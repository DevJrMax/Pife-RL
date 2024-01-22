import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .utils import *


class PifeMatch():
  def __init__(self, agents=[], logging_level=logging.INFO):
      self.total_n_cards = 104
      self.__n_cards_in_hand = 9
      self.__max_n_cards_in_hand = 10
      self.game_over = False
      self.winner = -1
      self.actual_player = -1
      self.actual_turn = 1
      self.n_games_for_win = 3
      self.n_shuffled_pile_in_deck = 0

      self.extra_actions = 3

      self.deck_game_ids = np.array([], dtype=np.int8)
      self.pile_game_ids = np.array([], dtype=np.int8)

      self.top_card_in_pile = -1

      self.players = {}

      self.agents = agents

      self.n_players = len(self.agents)

      self.deck_gui = generate_deck_gui()

      for agent in (self.agents):
        self.players[agent] = {
            'hand_ids': np.zeros((self.__n_cards_in_hand,), dtype=np.int8),
            'hand_gui_schema': {},
            'formed_games_ids': np.array([], dtype=np.int8),
            'known_cards_in_opponents_hand_ids': np.array([], dtype=np.int8),
            'unknown_cards_in_games':  np.array([], dtype=np.int8),
            'important_cards': np.array([], dtype=np.int8),
            'last_action': 0
        }

      # Set logger
      self.__logger = logging.getLogger()
      self.__logger.setLevel(logging_level)

      self.start_game()

  # Check games functions #
  def check_consecutive(self, to_check_list):
    return sorted(to_check_list) == list(range(min(to_check_list), max(to_check_list)+1))

  def check_identical(self, to_check_list):
    return all(i == to_check_list[0] for i in to_check_list)

  def check_unique(self, list):
    return not (len(list) > len(set(list)))

  def remove_games_from_hand(self, available_cards_ids, games):
    for game in games:
      available_cards_ids = [e for e in available_cards_ids if e not in game]

    return available_cards_ids, games

  def order_hand(self, hand, by:Order_hand_by=Order_hand_by.SUIT):
    if by.value == 'suit':
      hand = dict(sorted(hand.items(), key=lambda x: (x[1]['suit'], x[1]['value'])))
    else:
      hand = dict(sorted(hand.items(), key=lambda x: (x[1]['value'], x[1]['suit'])))
    return hand

  def check_game(self, combination, hand):
    is_a_game = False
    cards_in_combination = []

    for card_id in combination:
      cards_in_combination.append(hand[card_id])

    values = [card['value'] for card in cards_in_combination]
    suits = [card['suit'] for card in cards_in_combination]


    #Checking if are sequence of same suit
    if self.check_consecutive(values) and self.check_identical(suits):
      is_a_game = True
    #Checking if are same number of different suit
    elif self.check_identical(values) and self.check_unique(suits):
      is_a_game = True
    else:
      is_a_game = False

    return is_a_game

  def get_formed_games(self, hand, n_cards = 3):
    games = []
    orders = [Order_hand_by.SUIT, Order_hand_by.NUMBER]

    for order_by in orders:
      hand = self.order_hand(hand=hand, by=order_by)
      available_cards_ids = list(hand.keys())

      while True:
        # Removing games
        available_cards_ids, games = self.remove_games_from_hand(available_cards_ids, games)

        available_combinations = list(itertools.combinations(available_cards_ids, n_cards))
        has_game = False

        for combination in available_combinations:
          if self.check_game(combination, hand):
            has_game = True
            games.append(combination)
            break
          else:
            pass

        if not has_game:
          break
        else:
          pass

    formed_games =  []

    for index, game in enumerate(games, 1):
      game_array = np.array(list(game), dtype=np.int8)
      formed_games.append(game_array)

    formed_games = np.array(formed_games)

    return formed_games
  
  def get_important_cards(self, hand):
    possible_games = []

    hand = self.order_hand(hand=hand, by=Order_hand_by.SUIT)
    available_cards_ids = list(hand.keys())

    available_combinations = list(itertools.combinations(available_cards_ids, 2))

    for combination in available_combinations:
      if self.check_game(combination, hand):
        if combination not in possible_games:
          possible_games.append(combination)
    
    important_cards = list(sum(possible_games,()))

    important_cards = list(dict.fromkeys(important_cards))
    
    important_cards = np.array(list(important_cards), dtype=np.int8)

    return important_cards
  # Check games functions #

  def plot_hand(self, player_id = None, order=None):
      if not player_id:
        player_id = self.actual_player

      hand_gui_schema = self.players[player_id]['hand_gui_schema']
      if not order:
        hand_gui_schema = hand_gui_schema
      elif order.value == 'suit' or order.value == 'value':
        hand_gui_schema = self.order_hand(hand=hand_gui_schema, by=order)
      else:
        hand_gui_schema = hand_gui_schema

      fig = plt.figure(figsize=(16, 16))
      columns = len(self.players[player_id]['hand_gui_schema'])
      rows = 1

      cards_ids = list(hand_gui_schema.keys())

      for i in range(1, columns*rows +1):
        card_id = cards_ids[i-1]
        fig.add_subplot(rows, columns, i)
        img = mpimg.imread(hand_gui_schema[card_id]['card_image'])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
      plt.show()

  def ids_to_gui_schema(self, ids):
    gui_schema = {}
    for id in ids:
      gui_schema[id] = self.deck_gui[id]
    return gui_schema

  def order_gui_schema_by_games(self, player_id):
    formed_games_ids = self.players[player_id]['formed_games_ids']
    hand_ids = self.players[player_id]['hand_ids']

    if len(formed_games_ids) > 0:
      games = np.hstack(formed_games_ids)
      games = games.astype(np.int8)
    else:
      games = np.array([], dtype=np.int8)

    mask = np.isin(hand_ids, games)

    no_games_cards = hand_ids[~mask]

    games_ordered_ids = np.hstack([games, no_games_cards])
    games_ordered_ids = games_ordered_ids.astype(np.int8)

    hand_gui_schema_ordered = self.ids_to_gui_schema(games_ordered_ids)

    return hand_gui_schema_ordered

  def get_unknown_cards_in_games(self, agent_id):
    unknown_cards_in_games = np.arange(0, self.total_n_cards, 1, dtype=np.int8)
    unknown_cards_in_games = np.setdiff1d(unknown_cards_in_games, self.players[agent_id]['hand_ids'])
    unknown_cards_in_games = np.setdiff1d(unknown_cards_in_games, self.players[agent_id]['known_cards_in_opponents_hand_ids'])
    unknown_cards_in_games = np.setdiff1d(unknown_cards_in_games, self.pile_game_ids)

    if self.top_card_in_pile != -1:
      unknown_cards_in_games = np.setdiff1d(unknown_cards_in_games, np.array([self.top_card_in_pile]))

    return unknown_cards_in_games

  def check_game_over(self, player_id):
    n_formed_games = len(self.players[player_id]['formed_games_ids'])
    if n_formed_games == 3:
      self.winner = player_id
      self.game_over = True
    else:
      self.game_over = False

  def __next_player(self):
    self.actual_player = self.actual_player + 1

    if self.actual_player == self.n_players + 1:
      self.actual_player = 1

    self.actual_turn = self.actual_turn + 1

    self.__logger.info(f'Passing to turn: {self.actual_turn} - Player: {self.actual_player}')

  def __shuffle_pile_in_a_new_deck(self):
    self.n_shuffled_pile_in_deck = self.n_shuffled_pile_in_deck + 1
    self.deck_game_ids = self.pile_game_ids
    np.random.shuffle(self.deck_game_ids)

    self.pile_game_ids = np.array([], dtype=np.int8)

  def ids_to_one_hot_hand(self, card_ids):
      one_hot_cards = np.zeros((self.total_n_cards,), dtype=np.int8)
      one_hot_cards[card_ids] = np.ones((len(card_ids),), dtype=np.int8)
      return one_hot_cards

  def _update_player(self, player_id, new_hand_ids):
    self.players[player_id]['hand_ids'] = new_hand_ids
    self.players[player_id]['hand_gui_schema'] = self.ids_to_gui_schema(self.players[player_id]['hand_ids'])
    self.players[player_id]['formed_games_ids'] = self.get_formed_games(self.players[player_id]['hand_gui_schema'])
    self.players[player_id]['hand_gui_schema'] = self.order_gui_schema_by_games(player_id)
    self.players[player_id]['unknown_cards_in_games'] = self.get_unknown_cards_in_games(player_id)
    self.players[player_id]['important_cards'] = self.get_important_cards(self.players[player_id]['hand_gui_schema'])

  def start_game(self):
    # Reset game over status
    self.game_over = False

    # Reset the winner
    self.winner = -1

    # Reset first player
    self.actual_player = 1

    # Reset actual turn
    self.actual_turn = 1

    # Reset the deck
    self.deck_game_ids = np.arange(0, self.total_n_cards, 1, dtype=np.int8)

    # Reset the pile
    self.pile_game_ids = np.array([], dtype=np.int8)
    self.top_card_in_pile = -1

    # Shuffle the deck
    np.random.shuffle(self.deck_game_ids)

    players_hand_ids = []

    for id, agent_id in enumerate(self.agents):
      if agent_id == "player_1":
        players_hand_ids.append(self.deck_game_ids[0 : self.__n_cards_in_hand])
      else:
        players_hand_ids.append(self.deck_game_ids[id * self.__n_cards_in_hand : (id + 1) * self.__n_cards_in_hand])

    self.deck_game_ids = self.deck_game_ids[self.n_players * self.__n_cards_in_hand : ]

    # Attribute the the choosed ids for each players
    for id, agent_id in enumerate(self.agents):
      self._update_player(agent_id, players_hand_ids[id])

  def play_turn(self, agent_id, action):

    self.__logger.info(f'Turn: {self.actual_turn} Player: player_{self.actual_player} Action: {action}')
    next_turn = False
    n_cards_in_hand = len(self.players[agent_id]['hand_ids'])

    if agent_id != f'player_{self.actual_player}':
      self.__logger.debug(f'It\'s player {self.actual_player} turn!')
      return next_turn

    # Actions we can take:
    ## 0 - Get card from top of deck
    if action == 0:
      if(len(self.deck_game_ids) == 0):
        self.__logger.info('Deck is over. Shuffling the cards in pile in a new deck.')
        self.__shuffle_pile_in_a_new_deck()

      if n_cards_in_hand < self.__max_n_cards_in_hand:
        drawed_card, self.deck_game_ids = self.deck_game_ids[-1], self.deck_game_ids[:-1]
        new_hand_ids = np.append(self.players[agent_id]['hand_ids'], [drawed_card])
        self._update_player(agent_id, new_hand_ids)
        self.players[agent_id]['last_action'] = action

      else:

        self.__logger.debug(f'You have {n_cards_in_hand} in the hand you can\'t draw. Need discard one!')

    ## 1 - Get card from pile
    elif action == 1:
      if n_cards_in_hand == self.__max_n_cards_in_hand:
        self.__logger.debug(f'You have {n_cards_in_hand} in the hand you can\'t draw. Need discard one!')

      elif self.top_card_in_pile == -1:
        self.__logger.debug('Pile is empty. Need draw from deck!')

      elif n_cards_in_hand < self.__max_n_cards_in_hand:
        drawed_card = self.top_card_in_pile

        for index in range(1, self.n_players + 1):
          if agent_id != f'player_{index}':
            self.players[f'player_{index}']['known_cards_in_opponents_hand_ids'] = np.append(self.players[f'player_{index}']['known_cards_in_opponents_hand_ids'], [drawed_card])

        if len(self.pile_game_ids) == 0:
          self.top_card_in_pile = -1
        else:
          self.top_card_in_pile = self.pile_game_ids[-1]
          self.pile_game_ids = self.pile_game_ids[:-1]

        new_hand_ids = np.append(self.players[agent_id]['hand_ids'], [drawed_card])
        self._update_player(agent_id, new_hand_ids)
        self.players[agent_id]['last_action'] = action

      else:
        pass

    ## 2 - Win the game (Have 3 valid games formed)
    elif action == 2:
      self.check_game_over(agent_id)

      if self.game_over:
        self.winner = agent_id
        self.__logger.critical(f'Congratulations player {agent_id} you win in turn {self.actual_turn}!! N. shuffle: {self.n_shuffled_pile_in_deck}. Deck remain: {len(self.deck_game_ids)}')
      else:
        self.__logger.critical(f'Player {agent_id} is not the winner. You have {len(self.players[agent_id]["formed_games_ids"])} game(s) formed!')

    ## 3 - Card to discard
    else:
      card_to_discard = action - self.extra_actions

      if n_cards_in_hand == self.__max_n_cards_in_hand:
        if np.any(self.players[agent_id]['hand_ids'] == card_to_discard):
          new_hand_ids = np.delete(self.players[agent_id]['hand_ids'], np.where(self.players[agent_id]['hand_ids'] == card_to_discard))
          if self.top_card_in_pile != -1:
            self.pile_game_ids = np.append(self.pile_game_ids, [self.top_card_in_pile])
          self.top_card_in_pile = card_to_discard

          for index in range(1, self.n_players + 1):
            if agent_id != f'player_{index}':
              self.players[f'player_{index}']['known_cards_in_opponents_hand_ids'] = np.delete(self.players[f'player_{index}']['known_cards_in_opponents_hand_ids'], np.where(self.players[f'player_{index}']['known_cards_in_opponents_hand_ids'] == card_to_discard))

          self._update_player(agent_id, new_hand_ids)

          for index in range(1, self.n_players + 1):
            self._update_player(f'player_{index}', self.players[f'player_{index}']['hand_ids'])
          self.players[agent_id]['last_action'] = action
          self.__next_player()
          next_turn = True
        else:
          self.__logger.debug('This card is not in your hand!')
      else:
        self.__logger.debug(f'You have {n_cards_in_hand} in the hand you can\'t discard. Need draw from the deck or from the pile!')

    return next_turn