from __future__ import annotations

import pygame
import logging
import gymnasium
import numpy as np
from .utils import *
from .pife_match import *
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pife_v0",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        screen_height: int | None = 900,
        n_players: int = 2,
        show_unknown_cards_in_game: bool = True,
        show_known_cards_in_opponents_hand: bool = True,
        add_formed_games: bool = True,
        max_turns: int = 500,
        freeze_formed_games: bool = False,
        logging_level: int = logging.CRITICAL,
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode, screen_height, n_players)
        self.agents = [f"player_{player_id}" for player_id in range(1, n_players + 1)]

        self.logging_level = logging_level

        self.pife_match = PifeMatch(self.agents, logging_level=self.logging_level)

        self.possible_agents = self.agents[:]

        self.max_turns = max_turns

        self.add_formed_games = add_formed_games

        self.show_known_cards_in_opponents_hand = show_known_cards_in_opponents_hand

        self.freeze_formed_games = freeze_formed_games

        self.show_unknown_cards_in_game = show_unknown_cards_in_game

        self.extra_actions = self.pife_match.extra_actions

        self.total_n_actions = self.pife_match.total_n_cards + self.extra_actions

        self.action_spaces = {agent_id: spaces.Discrete(self.total_n_actions) for agent_id in self.agents}

        num_features = 0

        if self.add_formed_games:
            num_features = 9
        else:
            num_features = 6

        if not self.show_unknown_cards_in_game:
            num_features = num_features - 1

        if not self.show_known_cards_in_opponents_hand:
            num_features = num_features - 1

        shape_obs = (num_features, self.pife_match.total_n_cards)

        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=shape_obs, dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(self.total_n_actions,), dtype=np.int8),
                }
            )
            for agent_id in self.agents
        }


        self.rewards = {agent_id: 0 for agent_id in self.agents}
        self.terminations = {agent_id: False for agent_id in self.agents}
        self.truncations = {agent_id: False for agent_id in self.agents}
        self.infos = {agent_id: {"legal_moves": list(range(0, self.total_n_actions))} for agent_id in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen_width = screen_height * 1.5
        self.screen = None

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()


    def observe(self, agent):
      observation = self._get_observation(agent)
      action_mask = self._get_legal_moves(agent)
      return {"observation": observation, "action_mask": action_mask}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_observation(self, agent):
        obs = []
        one_hot_hand_ids = self.pife_match.ids_to_one_hot_hand(self.pife_match.players[agent]['hand_ids'])
        obs.append(one_hot_hand_ids)

        if self.pife_match.top_card_in_pile == -1:
            one_hot_top_card_pile = self.pife_match.ids_to_one_hot_hand([])
        else:
            one_hot_top_card_pile = self.pife_match.ids_to_one_hot_hand([self.pife_match.top_card_in_pile])
        obs.append(one_hot_top_card_pile)

        one_hot_pile_game_ids = self.pife_match.ids_to_one_hot_hand(self.pife_match.pile_game_ids)
        obs.append(one_hot_pile_game_ids)

        if self.show_known_cards_in_opponents_hand:
            one_hot_known_cards_in_opponents_hand_ids = self.pife_match.ids_to_one_hot_hand(self.pife_match.players[agent]['known_cards_in_opponents_hand_ids'])
            obs.append(one_hot_known_cards_in_opponents_hand_ids)
        
        if self.show_unknown_cards_in_game:
            one_hot_unknown_cards_in_games = self.pife_match.ids_to_one_hot_hand(self.pife_match.players[agent]['unknown_cards_in_games'])
            obs.append(one_hot_unknown_cards_in_games)
        
        one_hot_important_cards_ids = self.pife_match.ids_to_one_hot_hand(self.pife_match.players[agent]['important_cards'])
        obs.append(one_hot_important_cards_ids)

        if self.add_formed_games:
            one_hot_game_1 = np.zeros(self.pife_match.total_n_cards, dtype=np.int8)
            one_hot_game_2 = np.zeros(self.pife_match.total_n_cards, dtype=np.int8)
            one_hot_game_3 = np.zeros(self.pife_match.total_n_cards, dtype=np.int8)

            n_formed_games = len(self.pife_match.players[agent]['formed_games_ids'])

            if n_formed_games >= 1:
                one_hot_game_1 = self.pife_match.ids_to_one_hot_hand(self.pife_match.players[agent]['formed_games_ids'][0])

            if n_formed_games >= 2:
                one_hot_game_2 = self.pife_match.ids_to_one_hot_hand(self.pife_match.players[agent]['formed_games_ids'][1])

            if n_formed_games >= 3:
                one_hot_game_3 = self.pife_match.ids_to_one_hot_hand(self.pife_match.players[agent]['formed_games_ids'][2])

            obs.append(one_hot_game_1)
            obs.append(one_hot_game_2)
            obs.append(one_hot_game_3)

        obs = np.array(obs, dtype=np.int8)

        return obs

    def _get_legal_moves(self, agent):
        legal_moves = np.zeros(self.total_n_actions, dtype=np.int8)
        n_cards_in_hands = len(self.pife_match.players[agent]['hand_ids'])
        n_formed_games = len(self.pife_match.players[agent]['formed_games_ids'])
        last_action = self.pife_match.players[agent]['last_action']

        if n_cards_in_hands == 9:
            legal_moves[0] = 1
            legal_moves[1] = 1
        else:
            legal_moves[self.pife_match.players[agent]['hand_ids'] + self.extra_actions] = np.ones(len(self.pife_match.players[agent]['hand_ids']), dtype=np.int8)

        if last_action == 1:
            legal_moves[self.pife_match.players[agent]['hand_ids'][-1:] + self.extra_actions] = 0

        if self.freeze_formed_games:
            for formed_game in self.pife_match.players[agent]['formed_games_ids']:
                legal_moves[np.array(formed_game, dtype=np.int8) + self.extra_actions] = 0 

        if n_formed_games == 3:
            legal_moves = np.zeros(self.total_n_actions, dtype=np.int8)
            legal_moves[2] = 1

        return legal_moves

    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        # check if input action is a valid move (0 == empty spot)
        assert self._get_legal_moves(self.agent_selection)[action] == 1, "played illegal move"

        reward = 0

        if action > 2:
            actual_discarded_card = action - self.extra_actions
            if actual_discarded_card in self.pife_match.players[self.agent_selection]['important_cards']:
                reward = -0.5
            else:
                reward = 0.5

        before_action_n_formed_games = len(self.pife_match.players[self.agent_selection]['formed_games_ids'])

        # play turn
        next_turn = self.pife_match.play_turn(self.agent_selection, action)

        after_action_n_formed_games = len(self.pife_match.players[self.agent_selection]['formed_games_ids'])      
        
        if before_action_n_formed_games < after_action_n_formed_games:
            reward = reward + 0.2
        elif before_action_n_formed_games > after_action_n_formed_games:
            reward = reward - 0.2

        self.rewards[self.agent_selection] = self.rewards[self.agent_selection] + reward

        if not self.pife_match.game_over:
            pass
        else:
            for agent in self.agents:
                if agent == self.pife_match.winner:
                    self.rewards[agent] = self.rewards[agent] + 1
                else:
                    self.rewards[agent] = self.rewards[agent] - 1
                # once either play wins or there is a draw, game over, both players are done
                self.terminations = {agent_id: True for agent_id in self.agents}

        if self.max_turns == self.pife_match.actual_turn:
            self.truncations = {agent_id: True for agent_id in self.agents}
            for agent in self.agents:
                    self.rewards[agent] = 0

        # Switch selection to next agents

        if next_turn:
            next_agent = self._agent_selector.next()
            self._cumulative_rewards[self.agent_selection] = 0 
            self.agent_selection = next_agent

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):

        # reset environment
        self.agents = self.possible_agents[:]

        self.pife_match = PifeMatch(self.agents, logging_level=self.logging_level)

        self.rewards = {agent_id: 0 for agent_id in self.agents}
        self._cumulative_rewards = {agent_id: 0 for agent_id in self.agents}
        self.terminations = {agent_id: False for agent_id in self.agents}
        self.truncations = {agent_id: False for agent_id in self.agents}
        self.infos = {agent_id: {} for agent_id in self.agents}

        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

        if self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Pife")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

    def close(self):
        pass

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return


        deck_gui = generate_deck_gui()

        # Initializing RGB Color
        color = (11, 163, 92)

        # Changing surface color
        self.screen.fill(color)

        # Displaying cards
        # Load images and set initial positions
        players_card_images = {}
        titles = []
        font1 = pygame.font.SysFont('comicsansms.ttf', 30)
        font2 = pygame.font.SysFont('comicsansms.ttf', 36)
        for agent_id in self.possible_agents:
            players_card_images[agent_id] = {
            'images': [],
            'positions': [],
            'rects': []
            }
            titles.append(font1.render(agent_id, True, (0, 0, 0)))

        spacing = 125
        for agent_id in self.possible_agents:
            # height_player = 100
            count = 0.2
            for card_id in self.pife_match.players[agent_id]['hand_gui_schema']:
                card_info = deck_gui[card_id]

                original_image = pygame.image.load(card_info['card_image'])

                new_size = (100, 145)
                resized_image = pygame.transform.scale(original_image, new_size)

                players_card_images[agent_id]['images'].append(
                    resized_image
                )

                if agent_id == 'player_1':
                    current_pos = (count * spacing, 700)
                else:
                    current_pos = (count * spacing, 50)

                players_card_images[agent_id]['positions'].append(
                    current_pos
                )
                count = count + 1
            
            players_card_images[agent_id]['rects'] = [image.get_rect(topleft=pos) for image, pos in zip(players_card_images[agent_id]['images'], players_card_images[agent_id]['positions'])]

            for agent_id, title in zip(self.possible_agents, titles):
                for image, rect in zip(players_card_images[agent_id]['images'], players_card_images[agent_id]['rects']):
                    self.screen.blit(image, rect.topleft)
                    if agent_id == 'player_1':
                        self.screen.blit(title, (self.screen_width/2, 670))
                        self.screen.blit(font2.render(f'Formed games: {len(self.pife_match.players[agent_id]["formed_games_ids"])}', True, (0, 0, 0)), (50, 670))

                    else:
                        self.screen.blit(title, (self.screen_width/2, 25))
                        self.screen.blit(font2.render(f'Formed games: {len(self.pife_match.players[agent_id]["formed_games_ids"])}', True, (0, 0, 0)), (50, 25))


        # Number of turns
        self.screen.blit(font2.render(f'Turn: {self.pife_match.actual_turn}', True, (0, 0, 0)), (50, self.screen_height/2))

        # Deck zone
        self.screen.blit(font2.render('Deck', True, (0, 0, 0)), (1100, self.screen_height/2 - 95))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(1100, self.screen_height/2 - 70, 105, 150),  2)
        if len(self.pife_match.deck_game_ids) > 0:
            top_card_id = self.pife_match.deck_game_ids[-1]
            card_color = deck_gui[top_card_id]['deck_color']
            if card_color == 0:
                original_image = pygame.image.load('./imgs/blue.jpg')
            else:
                original_image = pygame.image.load('./imgs/red.jpg')
            new_size = (100, 145)
            resized_image = pygame.transform.scale(original_image, new_size)
            self.screen.blit(resized_image, (1102, self.screen_height/2 - 68))

        # Pile zone
        self.screen.blit(font2.render('Pile', True, (0, 0, 0)), (550, self.screen_height/2 - 95))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(550, self.screen_height/2 - 70, 105, 150),  2)
        if self.pife_match.top_card_in_pile != -1:
            card_info = deck_gui[self.pife_match.top_card_in_pile]
            original_image = pygame.image.load(card_info['card_image'])
            new_size = (100, 145)
            resized_image = pygame.transform.scale(original_image, new_size)
            self.screen.blit(resized_image, (552, self.screen_height/2 - 68))

        # Winner:
        font3 = pygame.font.SysFont('comicsansms.ttf', 60)
        if self.pife_match.winner != -1:
            self.screen.blit(font3.render(f'Winner: {self.pife_match.winner}', True, (255, 0, 0)), (50, self.screen_height/2 + 50))

        pygame.display.flip()

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])