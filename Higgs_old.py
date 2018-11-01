#!/usr/bin/env python3
# Python 3.6
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import logging
from collections import defaultdict, OrderedDict, namedtuple


def get_path_halite_cost(start, end, map):
    cost = 0
    moves = map.get_unsafe_moves(start, end)
    curr = start
    # TODO: path should be optimised - direct isn't always the best, A*?
    while moves:
        cost += map[curr].halite_amount * .1
        # quick hack, if choose lowest then shouldn't use naive_navigate
        curr = curr.directional_offset(moves[0])
        moves = map.get_unsafe_moves(curr, end)
    return cost * 2  # round-trip


def get_halite_cells(map, me):
    full_coord = [map[Position(j, i)] for i in range(0, map.height) for j in range(0, map.width)]
    shipyard = me.shipyard.position
    full_coord.remove(map[shipyard])
    full_coord.sort(key=lambda x: x.halite_amount / map.calculate_distance(shipyard, x.position), reverse=False)
    # TODO: take into account that not all halite can be extracted from the cell.
    # TODO: add discount factor for turn
    #full_coord.sort(key=lambda x: (x.halite_amount - get_path_halite_cost(shipyard, x.position, map))
    #                              * .9 ** map.calculate_distance(shipyard, x.position), reverse=False)
    #for f in full_coord:
    #    logging.info("Pos (%s,%s) has %s halite and is %s far from shipyard, costs %s",
    #                 f.position.x, f.position.y, f.halite_amount, map.calculate_distance(shipyard, f.position),
    #                 get_path_halite_cost(shipyard, f.position, map))
    # TODO: store the corresponding path to follow
    return full_coord


def get_target_direction(source, target):
    return (Direction.South if target.y > source.y else Direction.North if target.y < source.y else None,
            Direction.East if target.x > source.x else Direction.West if target.x < source.x else None)


def get_possible_moves(target, ship, game_map):
    # No need to normalize destination, since get_unsafe_moves does that
    if ship.halite_amount >= .1 * game_map[ship.position].halite_amount:
        possible_moves = game_map.get_unsafe_moves(ship.position, target)
        # possible_moves = [m for m in possible_moves
        #                  if game_map[ship.position.directional_offset(m)].ship is None or
        #                  game_map[ship.position.directional_offset(m)].ship.id in me]
        possible_moves.sort(key=lambda x: game_map[ship.position.directional_offset(x)].halite_amount)
        return possible_moves
    else:
        logging.info("Ship {} has not enough halite to move anywhere.".format(ship.id))
        return []


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
            logging.info("Marked cell {}, ship = None".format(game_map[ship.position].ship))

    moves = [m for pos, m in path]
    for ship, move in zip(ships, moves):
        logging.info('{}{}'.format(ship, move))
        if ship and ship.id not in command_dict.keys():
            new_pos = ship.position.directional_offset(move)
            logging.info("Ship {} will move {} to {}".format(ship.id, move, new_pos))
            command_dict[ship.id] = ship.move(move)
            game_map[new_pos].mark_unsafe(ship)


def resolve_moves_recursive(path, ship_id, graph, command_dict, game_map):
    ship_plan = graph[ship_id]

    for t_pos, t_move in ship_plan.to:
        if not game_map[t_pos].is_occupied:
            # Path viable
            path.append((t_pos, t_move))
            logging.info("Path found: {}".format(path))
            execute_path(path, command_dict, game_map)
            return
        else:
            if game_map[t_pos].ship.id in command_dict.keys():
                return  # path is being used next turn (rmb reg move will clear cell so should hit line 138)

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
                return
            else:
                # keep searching, DFSs
                next_ship_id = game_map[t_pos].ship.id
                new_path = path.copy()
                new_path.append((t_pos, t_move))
                resolve_moves_recursive(new_path, next_ship_id, graph, command_dict, game_map)


def register_move(ship, move, command_dict, game_map):
    new_pos = ship.position.directional_offset(move)
    logging.info("Ship {} will move {} to {}".format(ship.id, move, new_pos))
    command_dict[ship.id] = ship.move(move)
    game_map[ship.position].ship = None
    game_map[new_pos].mark_unsafe(ship)


ShipPlan = namedtuple('ShipPlan', 'ship to')

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    graph = {}
    target_count = defaultdict(int)
    command_queue = []
    command_dict = {}
    nextpos_ship = defaultdict(list)  # Possible moves

    # 0. Setting targets
    logging.info("#0 Setting targets...")
    for ship in me.get_ships():
        # logging.info("Ship {} at {} has {} halite.".format(ship.id, ship.position, ship.halite_amount))
        # TODO: set condition to create new drop point
        # TODO: set colliding final drop off
        # TODO: try harassing late game.
        if game.turn_number == MAX_TURN - 35:  # it's late, ask everyone to come home
           ship_targets[ship.id] = me.shipyard.position

        if ship.id not in ship_targets:  # new ship - set navigation direction
            ship_targets[ship.id] = targets.pop()
        elif ship.position == ship_targets[ship.id]:
            # reached target position
            if ship_targets[ship.id] == me.shipyard.position:  # reached shipyard, assign new target
                ship_targets[ship.id] = targets.pop()
            else:  # reached halite deposit, back if position is depleted or ship full, else stay
                if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
                    # TODO: find nearest drop point, get drop-offs
                    ship_targets[ship.id] = me.shipyard.position
                else:  # Staying, mark cells
                    register_move(ship, Direction.Still, command_dict, game_map)

        logging.info("Ship {} at {} has target {} ".format(ship.id, ship.position, ship_targets[ship.id]))

    # 1. Finding potential moves (has energy, will include blocked for resolution)
    # TODO: account for enemies!
    logging.info("#1 Finding all potential moves...")
    for ship in me.get_ships():
        possible_moves = get_possible_moves(ship_targets[ship.id], ship, game_map)

        for m in possible_moves:
            p = ship.position.directional_offset(m)
            if not game_map[p].is_occupied:
                register_move(ship, m, command_dict, game_map)
                if pos_to_hash_key(p) in nextpos_ship.keys():
                    del nextpos_ship[pos_to_hash_key(p)]
                break
            elif game_map[p].ship.id in [s.id for s in me.get_ships()]:
                # one of our own ships that can be resolved
                nextpos_ship[pos_to_hash_key(p)].append(ship)
                target_count[pos_to_hash_key(p)] += 1
                logging.info("Ship {} can go {} to {}".format(ship.id, m, ship.position.directional_offset(m)))
            else:
                # (try to) crash enemy ship if it has more halite than us
                if game_map[p].ship.halite_amount > ship.halite_amount:
                    logging.info("Ship {} has found an enemy ship to crash!")
                    register_move(ship, m, command_dict, game_map)
                    if pos_to_hash_key(p) in nextpos_ship.keys():
                        del nextpos_ship[pos_to_hash_key(p)]
                    break

        if ship.id not in command_dict.keys():  # no free move
            pos_and_moves = [(ship.position.directional_offset(m), m) for m in possible_moves
                             if game_map[ship.position.directional_offset(m)].ship.id in
                             [s.id for s in me.get_ships()]]  # filter out enemy ships
            logging.info("Ship {} has {} possible moves".format(ship.id, len(possible_moves)))
            graph[ship.id] = ShipPlan(ship, pos_and_moves)

    # 2. Set Moves to free cells (unoccupied this turn, and only one ship is going next)
    logging.info(graph)

    # 3. Resolve possible collisions, starting from blocked cells
    # then by cells targeted by only one ship, (sort by number of ships)
    logging.info("#3 Resolving collisions...")
    graph_ordered = OrderedDict(
        sorted(graph.items(), key=lambda x: (len(x[1][1]),
                                             target_count[pos_to_hash_key(x[1][1][0][0] if len(x[1][1]) > 0 else None)])))
    logging.info("graph_ordered dictionary (need to resolve){}".format(graph_ordered))

    for key in graph_ordered.keys():
        resolve_moves_recursive([], key, graph_ordered, command_dict, game_map)

    # 4. Remaining ships are blocked.
    logging.info("#4 Register blocked ships...")

    # Queue up commands
    for k, v in command_dict.items():
        command_queue.append(v)

    if game.turn_number <= (MAX_TURN - 200) and me.halite_amount >= constants.SHIP_COST and \
            not game_map[me.shipyard].is_occupied and len(me.get_ships()) < 10:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
