#!/usr/bin/env python3
# Python 3.6
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import logging
from collections import defaultdict, OrderedDict


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
    full_coord.sort(key=lambda x: x.halite_amount / 2 / map.calculate_distance(shipyard, x.position), reverse=False)
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


def get_possible_moves(target, ship, tplus_occupied, game_map):
    # No need to normalize destination, since get_unsafe_moves does that
    if ship.halite_amount >= .1 * game_map[ship.position].halite_amount:
        possible_moves = game_map.get_unsafe_moves(ship.position, target)
        #possible_moves = [m for m in possible_moves
        #                  if pos_to_hash_key(ship.position.directional_offset(m)) not in tplus_occupied]
        possible_moves.sort(key=lambda x: game_map[ship.position.directional_offset(x)].halite_amount)
        return possible_moves
    else:
        logging.info("Ship {} has not enough halite to move anywhere.".format(ship.id))
        return []


def get_safe_direction(target, ship, game_map):
    # Returns direction only if target isn't occupied (or stay), will then mark origin safe
    # Prioritises the direction that optimises Halite cost
    # Returns None if no safe Direction
    possible_moves = get_possible_moves(target, ship, [], game_map)
    if len(possible_moves) == 0:
        return Direction.Still

    for direction in possible_moves:
        target_pos = ship.position.directional_offset(direction)
        if not game_map[target_pos].is_occupied:
            game_map[target_pos].mark_unsafe(ship)
            game_map[ship.position].mark_unsafe(None)
            return direction

    return None


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
    return str(position.x) + ',' + str(position.y)


def hash_key_to_pos(key):
    xy = key.split(',')
    return Position(int(xy[0]), int(xy[1]))


def block_moves_recursive(target_pos_key, nextpos_ship_ordered, ship_nextpos,
                          tplus_occupied, ship_moves, game_map):
    if target_pos_key not in nextpos_ship_ordered.keys():
        # no one will be blocked by target_pos
        return
    else:
        target_pos = hash_key_to_pos(target_pos_key)
        ships = nextpos_ship_ordered[target_pos_key]
        logging.info("{} is blocked, removing from dict".format(target_pos))
        del nextpos_ship_ordered[target_pos_key]
        # recursively block all ships that has only one move to here
        # these ships need to stay at where they are
        completely_blocked_ships = [s for s in ships if len(ship_nextpos[s.id]) <= 1]
        for s in completely_blocked_ships:
            key = pos_to_hash_key(s.position)
            tplus_occupied[key] = s
            ship_moves[this_ship.id] = Direction.Still
            logging.info("Ship {} is completely blocked, will stay at {}".format(this_ship.id, s.position))
            block_moves_recursive(key, nextpos_ship_ordered, ship_nextpos,
                                  tplus_occupied, ship_moves, game_map)


def resolve_moves_recursive(target_pos_key, prev_pos_key, nextpos_ship_ordered, ship_moves, game_map):
    # TODO: this routine should save a chain of positions/moves, and only assign moves when chain is resolved
    if target_pos_key not in nextpos_ship_ordered.keys():
        # this chain has been resolved
        return
    else:
        target_pos = hash_key_to_pos(target_pos_key)
        ships = nextpos_ship_ordered[target_pos_key]
        this_ship = None
        is_swap = False
        for s in ships:
            if pos_to_hash_key(s.position) == prev_pos_key:
                # need to swap with a specific one
                this_ship = s
                is_swap = True
        if this_ship is None:  # otherwise don't care, pick one that has not got move yet
            for s in ships:
                if s.id not in ship_moves.keys():
                    this_ship = s

        del nextpos_ship_ordered[target_pos_key]
        if this_ship is None:  # ships got other options, this position is suboptimal
            return

        this_pos = this_ship.position
        offset = target_pos - this_pos

        ship_moves[this_ship.id] = (offset.x, offset.y)
        logging.info("Ship {} will move {}".format(this_ship.id, target_pos))
        this_pos_key = pos_to_hash_key(this_pos)
        if not is_swap:
            resolve_moves_recursive(this_pos_key, target_pos_key, nextpos_ship_ordered, ship_moves, game_map)


while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    command_queue = []
    ship_moves = {}  # Move for next turn
    ship_nextpos = {}  # Possible moves (unsafe)
    tplus_occupied = {}
    nextpos_ship = defaultdict(list)  # Possible moves

    # 0. Setting targets
    logging.info("#0 Setting targets...")
    for ship in me.get_ships():
        #logging.info("Ship {} at {} has {} halite.".format(ship.id, ship.position, ship.halite_amount))
        # TODO: set condition to create new drop point
        # TODO: set colliding final drop off
        # TODO: try harassing late game.
        # if game.turn_number == 400:  # it's late, ask everyone to come home
        #   ship_targets[ship.id] = me.shipyard.position

        if ship.id not in ship_targets:  # new ship - set navigation direction
            ship_targets[ship.id] = targets.pop()
        elif ship.position == ship_targets[ship.id]:
            # reached target position
            if ship_targets[ship.id] == me.shipyard.position:  # reached shipyard, assign new target
                ship_targets[ship.id] = targets.pop()
            else:  # reached halite deposit, back if position is depleted or ship full, else stay
                if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
                    # TODO: find nearest drop point, get_dropoffs
                    ship_targets[ship.id] = me.shipyard.position
                else:  # Staying, mark cells
                    tplus_occupied[pos_to_hash_key(ship.position)] = ship
                    ship_moves[ship.id] = Direction.Still

        logging.info("Ship {} at {} has target {} ".format(ship.id, ship.position, ship_targets[ship.id]))

    # 1. Finding potential moves (has energy, will include blocked for resolution) TODO: account for enemies!
    logging.info("#1 Finding all potential moves...")
    for ship in me.get_ships():
        possible_moves = get_possible_moves(ship_targets[ship.id], ship, tplus_occupied, game_map)
        ship_nextpos[ship.id] = possible_moves
        logging.info("Ship {} has {} possible moves".format(ship.id, len(possible_moves)))
        if len(possible_moves) == 0:
            tplus_occupied[pos_to_hash_key(ship.position)] = ship
            ship_moves[ship.id] = Direction.Still
            continue
        for m in possible_moves:
            p = ship.position.directional_offset(m)
            nextpos_ship[pos_to_hash_key(p)].append(ship)
            logging.info("Ship {} can go {} to {}".format(ship.id, m, ship.position.directional_offset(m)))
    logging.info("nextpos_ship dictionary {}".format(nextpos_ship))

    # 2. Set Moves to free cells (unoccupied this turn, and only one ship is going next)
    freed_ship = []
    logging.info("#2 Set moves to free cells...")
    for pos_key, ships in nextpos_ship.items():
        pos = hash_key_to_pos(pos_key)
        if len(ships) == 1 and not game_map[pos].is_occupied:
            this_ship = ships[0]
            if this_ship.id in freed_ship:
                continue
            tplus_occupied[pos_key] = this_ship
            offset = pos - this_ship.position
            ship_moves[this_ship.id] = (offset.x, offset.y)
            freed_ship.append(this_ship.id)
            game_map[this_ship.position].ship = None
            logging.info("Ship {} is free to go {} to {}".format(
                this_ship.id, ship_moves[this_ship.id], this_ship.position.directional_offset(ship_moves[this_ship.id])))

    # 3. Resolve possible collisions, starting from blocked cells
    # then by cells targeted by only one ship, (sort by number of ships)
    logging.info("#3 Resolving collisions...")
    # Remove freed ships
    for pos_key, ships in nextpos_ship.items():
        ships = [s for s in ships if s.id not in freed_ship]
    nextpos_ship_to_resolve = {k: v for k, v in nextpos_ship.items() if len(v) > 0}
    nextpos_ship_ordered = OrderedDict(sorted(nextpos_ship_to_resolve.items(),
                                              key=lambda x: (len(x[1]), x[1][0].id)))
    logging.info("nextpos_ship_ordered dictionary (need to resolve){}".format(nextpos_ship_ordered))

    ## Work backwards from the blocked positions to see if cells has to be blocked
    blocked_pos = [me.get_ship(s_id).position.directional_offset(m) for s_id, m in ship_moves.items()
                   if m == Direction.Still]
    for pos in blocked_pos:
        pos_key = pos_to_hash_key(pos)
        block_moves_recursive(pos_key, nextpos_ship_ordered, ship_nextpos,
                              tplus_occupied, ship_moves, game_map)

    while len(nextpos_ship_ordered) > 0:
        first_pos_key, first_pos_ships = list(nextpos_ship_ordered.items())[0]  # TODO:slow
        resolve_moves_recursive(first_pos_key, None, nextpos_ship_ordered, ship_moves, game_map)

    # 4. Remaining ships are blocked.
    logging.info("#4 Register blocked ships...")
    for ship in [s for s in me.get_ships() if s.id not in ship_moves.keys()]:
        ship_moves[ship.id] = Direction.Still
        logging.info("Ship {} will stay still at {}".format(ship.id, ship.position))

    # Queue up commands
    for ship in me.get_ships():
        move = ship_moves[ship.id]
        game_map[ship.position.directional_offset(move)].mark_unsafe(ship)
        command_queue.append(ship.move(move))

    if game.turn_number <= (MAX_TURN - 200) and me.halite_amount >= constants.SHIP_COST and \
            not game_map[me.shipyard].is_occupied and len(me.get_ships()) < 7:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
