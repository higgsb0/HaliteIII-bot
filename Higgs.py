#!/usr/bin/env python3
# Python 3.6
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import logging
from collections import defaultdict, OrderedDict, namedtuple

# import numpy as np

global game


def get_path_halite_cost(start, end, map):
    cost = 0
    moves = map.get_unsafe_moves(start, end)
    curr = start
    # TODO: path should be optimised - direct isn't always the best, A*?
    while moves:
        cost += map[curr].halite_amount * .1
        moves.sort(key=lambda m: map[curr.directional_offset(m)].halite_amount)
        # assume we also travel along the path with least halite
        curr = map.normalize(curr.directional_offset(moves[0]))
        moves = map.get_unsafe_moves(curr, end)
    return cost


def get_halite_cells(map, me, descending=False):
    full_coord = [map[Position(j, i)] for i in range(0, map.height) for j in range(0, map.width)]
    dropoffs = get_dropoff_positions(me)
    for dropoff in dropoffs:
        full_coord.remove(map[dropoff])
    # Remember multiplying constant should have no effect at all!
    # Discount factor seems to worsen the results..
    # if game.turn_number < 250:
    #    full_coord.sort(key=lambda x: x.halite_amount * .97 ** (2 * map.calculate_distance(dropoff, x.position)), reverse=False)
    # else:
    full_coord.sort(key=lambda x: x.halite_amount /
                    (map.calculate_distance(get_nearest_dropoff(map, me, x.position), x.position)),
                    reverse=descending)
    # TODO: This doesn't quite work
    # full_coord.sort(key=lambda x: (x.halite_amount - get_path_halite_cost(dropoff, x.position, map))
    #                * .98 ** (2 * map.calculate_distance(dropoff, x.position)), reverse=False)
    return full_coord


def get_halite_cells_nearby(map, current_pos, radius=3, descending=False):
    full_coord = [map[game_map.normalize(Position(i, j))]
                  for i in range(current_pos.x - radius, current_pos.x + radius)
                  for j in range(current_pos.y - radius, current_pos.y + radius)]
    # if map[current_pos] in full_coord:
    #    full_coord.remove(map[current_pos])
    full_coord.sort(key=lambda x: x.halite_amount, reverse=descending)
    # TODO: distance/discount factor
    # full_coord.sort(key=lambda x: x.halite_amount * .98 ** map.calculate_distance(current_pos, x.position),
    #                reverse=False)

    return full_coord


def get_surrounding_halite(map, current_pos, radius=6):
    coords = [map[game_map.normalize(Position(i, j))]
              for i in range(current_pos.x - radius, current_pos.x + radius)
              for j in range(current_pos.y - radius, current_pos.y + radius)]

    total = sum([c.halite_amount for c in coords])
    return total


def get_target_direction(source, target):
    return (Direction.South if target.y > source.y else Direction.North if target.y < source.y else None,
            Direction.East if target.x > source.x else Direction.West if target.x < source.x else None)


def get_possible_moves(target, ship, game_map):
    # No need to normalize destination, since get_unsafe_moves does that
    if ship.halite_amount >= .1 * game_map[ship.position].halite_amount:
        possible_moves = game_map.get_unsafe_moves(ship.position, target)
        possible_moves.sort(key=lambda x: game_map[ship.position.directional_offset(x)].halite_amount)
        return possible_moves
    else:
        logging.info("Ship {} has not enough halite to move anywhere.".format(ship.id))
        return []


def get_dropoff_positions(me):
    dropoffs = [d.position for d in me.get_dropoffs()]
    dropoffs.append(me.shipyard.position)
    return dropoffs


def get_nearest_dropoff(map, me, position):
    dropoffs = sorted(get_dropoff_positions(me), key=lambda x: map.calculate_distance(position, x))
    return dropoffs[0]


def pos_to_hash_key(position):
    if position:
        return str(position.x) + ',' + str(position.y)
    else:
        return None


def hash_key_to_pos(key):
    xy = key.split(',')
    return Position(int(xy[0]), int(xy[1]))


def execute_path(path, command_dict, game_map):
    logging.info("Executing path: {}".format(path))
    ships = [game_map[pos.directional_offset(Direction.invert(move))].ship
             for pos, move in path]
    for ship in ships:
        if ship:
            game_map[ship.position].ship = None

    moves = [m for pos, m in path]
    for ship, move in zip(ships, moves):
        logging.info('{}{}'.format(ship, move))
        if ship and ship.id not in command_dict:  # TODO: (allow override if it's better)
            new_pos = ship.position.directional_offset(move)
            logging.info("Ship {} will move {} to {}".format(ship.id, move, new_pos))
            command_dict[ship.id] = ship.move(move)
            game_map[new_pos].mark_unsafe(ship)


def resolve_moves_recursive(path, ship_id, graph, command_dict, game_map):
    if ship_id in command_dict:
        return True  # ship has planned move already

    ship_plan = graph[ship_id]
    for t_pos, t_move in ship_plan.to:
        if not game_map[t_pos].is_occupied:
            # Path viable
            path.append((t_pos, t_move))
            logging.info("Path found: {}".format(path))
            execute_path(path, command_dict, game_map)
            return True

    # Hacking to ensure we use free cells first
    for t_pos, t_move in ship_plan.to:
        if game_map[t_pos].ship.id in command_dict:
            return False  # path is being used next turn (rmb reg move will clear cell so should hit line 138)

        if pos_to_hash_key(t_pos) in [pos_to_hash_key(p[0]) for p in path]:
            # encountered cycle, execute the cycle only, scrap the remaining chain
            logging.info("original path {}".format(path))
            path.append((t_pos, t_move))
            logging.info("appending {} and {} as the last step of path".format(t_pos, t_move))
            cycle_start_idx = next(i for i, p in enumerate(path) if pos_to_hash_key(p[0]) == pos_to_hash_key(t_pos))
            for x in range(0, cycle_start_idx + 1):
                path.pop(0)
                logging.info("popped at{}".format(x))
            logging.info("Cyclic Path found: {}".format(path))
            execute_path(path, command_dict, game_map)
            return True
        else:
            # keep searching, DFSs
            next_ship_id = game_map[t_pos].ship.id
            new_path = path.copy()
            new_path.append((t_pos, t_move))
            found = resolve_moves_recursive(new_path, next_ship_id, graph, command_dict, game_map)
            if found:
                return True


def register_move(ship, move, command_dict, game_map):
    new_pos = ship.position.directional_offset(move)
    logging.info("Ship {} will move {} to {}".format(ship.id, move, new_pos))
    command_dict[ship.id] = ship.move(move)
    game_map[ship.position].ship = None
    game_map[new_pos].mark_unsafe(ship)


mapsize_turn = {
    32: 401,
    40: 426,
    48: 451,
    56: 476,
    64: 501
}

ship_targets = {}  # Final goal of the ship

game = hlt.Game()
MAX_TURN = mapsize_turn[game.game_map.height]
targets = [c.position for c in get_halite_cells(game.game_map, game.me)]

game.ready("Higgs")
logging.info('Bot: Higgs.')

ShipPlan = namedtuple('ShipPlan', 'ship to')
ship_to_be_dropoff = None
is_4p = len(game.players) == 4

logging.info(is_4p)

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    graph = {}
    target_count = defaultdict(int)
    command_queue = []
    command_dict = {}
    nextpos_ship = defaultdict(list)  # Possible moves
    building_dropoff = False

    logging.info('Clearing targets of crashed ships')
    if me.get_ships() is not None:
        crashed_ships = [s_id for s_id in ship_targets if s_id not in [s.id for s in me.get_ships()]]
        for cs in crashed_ships:
            logging.info("Removing Ship {}".format(cs))
            del ship_targets[cs]
        if ship_to_be_dropoff and ship_to_be_dropoff not in [s.id for s in me.get_ships()]:
            logging.info("Ship {} was a dropoff candidate but has sunk!".format(ship_to_be_dropoff))
            ship_to_be_dropoff = None

    # TODO: maybe not all ships should blindly follow top targets
    targets = [c.position for c in get_halite_cells(game.game_map, game.me)
               if c.position not in ship_targets.values()]

    # 0. Setting targets
    logging.info("#0 Setting targets...")
    for ship in me.get_ships():
        # logging.info("Ship {} at {} has {} halite.".format(ship.id, ship.position, ship.halite_amount))
        if ship.id == ship_to_be_dropoff:  # wait till enough halite
            continue

        # TODO: try harassing late game.
        if ship.id not in ship_targets:  # new ship - set navigation direction
            ship_targets[ship.id] = targets.pop()

        nearest_dropoff = get_nearest_dropoff(game_map, me, ship.position)
        dist = game_map.calculate_distance(ship.position, nearest_dropoff)
        if MAX_TURN - game.turn_number - 10 < dist:
            # it's late, ask everyone to come home
            ship_targets[ship.id] = nearest_dropoff
            if dist == 1:  # CRASH
                move = game_map.get_unsafe_moves(ship.position, nearest_dropoff)[0]
                register_move(ship, move, command_dict, game_map)
        elif ship.position == ship_targets[ship.id]:
            # reached target position
            surrounding_halite = get_surrounding_halite(game_map, ship.position)
            dist_from_home = game_map.calculate_distance(ship.position, nearest_dropoff)
            if surrounding_halite > 3500 and dist_from_home > 20 and \
                    len(me.get_dropoffs()) < 1 and ship_to_be_dropoff is None \
                    and game.turn_number <= (MAX_TURN - 250):
                ship_to_be_dropoff = ship.id
                if ship.id in ship_targets:
                    del ship_targets[ship.id]
                logging.info("Ship {} is a candidate dropoff at {}".format(ship.id, ship.position))
                continue
            elif ship.position in get_dropoff_positions(me):  # reached shipyard, assign new target
                ship_targets[ship.id] = targets.pop()
            elif ship.is_full:
                ship_targets[ship.id] = get_nearest_dropoff(game_map, me, ship.position)
            elif game_map[ship.position].halite_amount < 50:
                new_targets = get_halite_cells_nearby(game_map, ship.position, descending=True)
                new_target = next(t.position for t in new_targets if t.position not in ship_targets.values())
                dist_from_home = game_map.calculate_distance(ship.position, nearest_dropoff)
                dist_from_target = game_map.calculate_distance(ship.position, new_target)
                # If Halite/turn of going to nearby target > going back then go to new target
                new_reward = ship.halite_amount + game_map[new_target].halite_amount - (
                    100 if game.turn_number < 350 else 50) - \
                             get_path_halite_cost(ship.position, new_target, game_map)
                if (ship.halite_amount * .98 ** (dist_from_home if game.turn_number < 250 else 0) <
                        min(1000, new_reward) * .98 ** (
                        (dist_from_home + dist_from_target) if game.turn_number < 250 else 0)):
                    logging.info('Ship {} finished collecting at {}, moving to nearby target {}'
                                 .format(ship.id, ship.position, new_target))
                    ship_targets[ship.id] = new_target
                else:
                    ship_targets[ship.id] = nearest_dropoff
            else:  # keep collecting
                register_move(ship, Direction.Still, command_dict, game_map)
        else:
            if ship.is_full:
                ship_targets[ship.id] = nearest_dropoff
            else:
                if ship_targets[ship.id] in get_dropoff_positions(me):
                    if ship.halite_amount < constants.MAX_HALITE * .7 and game.turn_number > 100:
                        # TODO: refactor this part, used above
                        logging.info("Mid game: Ship {} researching leftovers around{}".format(ship.id, ship.position))
                        new_targets = get_halite_cells_nearby(game_map, ship.position, descending=True)
                        new_target = next(t.position for t in new_targets if t.position not in ship_targets.values())
                        dist_from_home = game_map.calculate_distance(ship.position, nearest_dropoff)
                        dist_from_target = game_map.calculate_distance(ship.position, new_target)
                        # If Halite/turn of going to nearby target > going back then go to new target
                        new_reward = ship.halite_amount + game_map[new_target].halite_amount - (
                            100 if game.turn_number < 350 else 50) - \
                                     get_path_halite_cost(ship.position, new_target, game_map)
                        if (ship.halite_amount * .98 ** (dist_from_home if game.turn_number < 250 else 0) <
                                min(1000, new_reward) * .98 ** (
                                        (dist_from_home + dist_from_target) if game.turn_number < 250 else 0)):
                            logging.info('Ship {} altering its plan to nearby target {}'
                                         .format(ship.id, ship.position, new_target))
                            ship_targets[ship.id] = new_target
                            # TODO: can have infinite recursion
                elif game.turn_number > 350 and game_map[ship.position].halite_amount > 100:
                    logging.info("LATE GAME: Ship {} stalling at {}".format(ship.id, ship.position))
                    register_move(ship, Direction.Still, command_dict, game_map)
                elif game.turn_number < 250 and game_map[ship_targets[ship.id]].halite_amount < 100:
                    # TODO: this can be more deterministic, maybe memorize what the target was, or even calc EV
                    logging.info("Early game, Ship {}'s target at {} seems to have depleted, reassigning target"
                                 .format(ship.id, ship_targets[ship.id]))
                    ship_targets[ship.id] = targets.pop()
                else:
                    # traveling to target, check if current cell is too occupied
                    #diff_target_current = game_map[ship_targets[ship.id]].halite_amount - game_map[ship.position].halite_amount
                    #if diff_target_current > 0 and diff_target_current * \
                    #    .25 < .4 * get_path_halite_cost(ship.position, ship_targets[ship.id], game_map):
                    if game_map[ship.position].halite_amount > 500:
                        # more beneficial to stay than travel
                        logging.info("Ship {} finds it more beneficial to stall for one round".format(ship.id))
                        register_move(ship, Direction.Still, command_dict, game_map)
        logging.info("Ship {} at {} has target {} ".format(ship.id, ship.position, ship_targets[ship.id]))

    # 1. Finding potential moves (has energy, will include blocked for resolution)
    # TODO: account for enemies!
    logging.info("#1 Finding all potential moves...")
    for ship in me.get_ships():
        if ship.id == ship_to_be_dropoff:
            if ship.halite_amount + game_map[ship.position].halite_amount + me.halite_amount > 4000:
                command_dict[ship.id] = ship.make_dropoff()
                ship_to_be_dropoff = None
                building_dropoff = True
                logging.info("Ship {} will be converted to a dropoff at {}".format(ship.id, ship.position))
                continue
            else:  # wait till we have enough halite
                register_move(ship, Direction.Still, command_dict, game_map)
                continue

        possible_moves = get_possible_moves(ship_targets[ship.id], ship, game_map)
        if ship.position == me.shipyard.position and game.turn_number < 6:
            possible_moves.extend([d for d in Direction.get_all_cardinals()
                                   if d not in possible_moves
                                   and not game_map[ship.position.directional_offset(d)].is_occupied])

        for m in possible_moves:
            p = ship.position.directional_offset(m)
            if not game_map[p].is_occupied:
                register_move(ship, m, command_dict, game_map)
                if pos_to_hash_key(p) in nextpos_ship:
                    del nextpos_ship[pos_to_hash_key(p)]
                break
            elif game_map[p].ship.id in [s.id for s in me.get_ships()]:
                # one of our own ships that can be resolved
                nextpos_ship[pos_to_hash_key(p)].append(ship)
                target_count[pos_to_hash_key(p)] += 1
                logging.info("Ship {} can go {} to {}".format(ship.id, m, ship.position.directional_offset(m)))
            else:
                if is_4p:
                    continue  # not beneficial to crash in multi-players
                elif game_map[p].ship.halite_amount > ship.halite_amount or p in get_dropoff_positions(me):
                    # (try to) crash enemy ship if it has more halite than us or occupying our shipyard
                    logging.info("Ship {} has found an enemy ship to crash!")
                    register_move(ship, m, command_dict, game_map)
                    if pos_to_hash_key(p) in nextpos_ship:
                        del nextpos_ship[pos_to_hash_key(p)]
                    break
                # TODO: doesn't work!
                # else:
                #     logging.info("Ship {} is fleeing from enemy!")
                #     inv_m = Direction.invert(m)
                #     inv_p = ship.position.directional_offset(inv_m)
                #     if not game_map[inv_p].is_occupied:
                #         register_move(ship, inv_m, command_dict, game_map)
                #     if pos_to_hash_key(inv_p) in nextpos_ship:
                #         del nextpos_ship[pos_to_hash_key(inv_p)]
                #     break

        if ship.id not in command_dict:  # no free move
            pos_and_moves = [(ship.position.directional_offset(m), m) for m in possible_moves
                             if game_map[ship.position.directional_offset(m)].ship.id in
                             [s.id for s in me.get_ships()]]  # filter out enemy ships
            logging.info("Ship {} has {} possible moves".format(ship.id, len(possible_moves)))
            graph[ship.id] = ShipPlan(ship, pos_and_moves)

    # 2. Resolve possible collisions, starting from blocked cells
    # then by cells targeted by only one ship, (sort by number of ships)
    logging.info("#2 Resolving moves...")
    graph_ordered = OrderedDict(
        sorted(graph.items(), key=lambda x: (len(x[1][1]),
                                             target_count[
                                                 pos_to_hash_key(x[1][1][0][0] if len(x[1][1]) > 0 else None)])))
    logging.info("graph_ordered dictionary {}".format(graph_ordered))

    for key in graph_ordered:
        resolve_moves_recursive([], key, graph_ordered, command_dict, game_map)

    # Queue up commands
    for k, v in command_dict.items():
        command_queue.append(v)

    if game.turn_number <= (MAX_TURN - 200) and \
            me.halite_amount - (4000 if building_dropoff else 0) >= constants.SHIP_COST and \
            not game_map[me.shipyard].is_occupied and ship_to_be_dropoff is None \
            and len(me.get_ships()) < game_map.height / 2:
        # TODO: ship limit dependent on halite available on the maps
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
