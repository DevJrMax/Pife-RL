import enum

map_cards_dict = {0: 'ace', 10: 'jack', 11: 'queen', 12: 'king'}
map_suit_names_dict = {0: 'spades', 1: 'hearts', 2: 'diamonds', 3: 'clubs'}

class Order_hand_by(enum.Enum):
  SUIT = 'suit'
  NUMBER = 'value'

def generate_deck_gui():
    n_decks = 2
    n_cards = 52
    suits = ["♠", "♥", "♦", "♣"]
    deck = {}
    card_count = 0
    for deck_color in range(n_decks):
        for index_suit, suit in enumerate(suits):
            suit_name = map_suit_names_dict[index_suit]

            for card_value in range(n_cards//len(suits)):
                if map_cards_dict.get(card_value) != None:
                    card_name = map_cards_dict[card_value]
                else:
                    card_name = card_value + 1

                has_new_version = '2' if card_name in ['queen', 'king', 'jack'] else ''

                deck[card_count] = {
                    'deck_color': deck_color,
                    'suit': index_suit,
                    'value': card_value,
                    "card_image": f'./PNG-cards-1.3/{card_name}_of_{suit_name}{has_new_version}.png'
                }
                card_count = card_count + 1
    return deck