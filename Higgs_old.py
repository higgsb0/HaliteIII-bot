#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

def get_halite_cells(map, me):
    full_coord = [map[Position(j, i)] for i in range(0, map.height) for j in range(0, map.width)]
    shipyard = me.shipyard.position
    full_coord.remove(map[shipyard])
    full_coord.sort(key=lambda x: x.halite_amount / 2 / map.calculate_distance(shipyard, x.position), reverse=False)
    for f in full_coord:
        logging.info("Pos (%s,%s) has %s halite and is %s far from shipyard",
                     f.position.x, f.position.y, f.halite_amount, map.calculate_distance(shipyard, f.position))
    return full_coord


def try_navigate(ship, game_map):
    cells = [p for p in ship.position.get_surrounding_cardinals()]
    cells[1], cells[2] = cells[2], cells[1]
    unoccupied_pos = [c for c in cells if not game_map[c].is_occupied]

    if unoccupied_pos:
        logging.info("Ship {} path is blocked.".format(ship.id))
        return game_map.naive_navigate(ship, unoccupied_pos[0])
    return Direction.Still


map_size = {
    32: 401,
    40: 426,
    48: 451,
    56: 476,
    64: 501
}
ship_targets = {}

game = hlt.Game()
targets = [c.position for c in get_halite_cells(game.game_map, game.me)]
max_turn = map_size[game.game_map.width]
game.ready("Higgs benchmark")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info('Bot: Higgs.')

while True:
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    command_queue = []

    for ship in me.get_ships():
        logging.info("Ship {} has {} halite.".format(ship.id, ship.halite_amount))

        if ship.id not in ship_targets:
            # new ship - set navigation direction
            ship_targets[ship.id] = targets.pop()
            move = game_map.naive_navigate(ship, ship_targets[ship.id])
            if move is Direction.Still:
                move = try_navigate(ship, game_map)
        elif ship.position != ship_targets[ship.id]:
            move = game_map.naive_navigate(ship, ship_targets[ship.id])
            if move is Direction.Still:
                move = try_navigate(ship, game_map)
        else:
            # reached target position
            if ship_targets[ship.id] == me.shipyard.position:
                # reached shipyard, assign new target
                ship_targets[ship.id] = targets.pop()
                move = game_map.naive_navigate(ship, ship_targets[ship.id])
                # unstuck
                if move is Direction.Still:
                    move = try_navigate(ship, game_map)

            else:
                # reached halite deposit, back if position is depleted or ship full, else stay
                if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
                    ship_targets[ship.id] = me.shipyard.position
                    move = game_map.naive_navigate(ship, ship_targets[ship.id])
                    if move is Direction.Still:
                        move = try_navigate(ship, game_map)
                else:
                    # This is always Still
                    move = game_map.naive_navigate(ship, ship_targets[ship.id])

        logging.info("Ship {} with target {} is moving to {}.".format(ship.id, ship_targets[ship.id],
                                                                      ship.position.directional_offset(move)))
        command_queue.append(ship.move(move))

    if game.turn_number <= (max_turn - 200) and me.halite_amount >= constants.SHIP_COST and not game_map[
        me.shipyard].is_occupied \
            and len(me.get_ships()) < 9:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

