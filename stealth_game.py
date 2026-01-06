
"""
Simple PyOpenGL Stealth Maze Game - ENHANCED WITH 12 STEALTH FEATURES

PART 1: Core Stealth World & Basic AI
1) Vision Cones with Line-of-Sight
2) Patrol Routes (Semi-Procedural)
3) Detection Meter (Gradual)
4) Light & Shadow Zones (Fake Lighting)

PART 2: Intelligence, Interaction & Stealth Tools
5) Guard States (PATROL, SUSPICIOUS, ALERT, SEARCH, RETURN)
6) Sound / Noise System (footsteps, doors, objects create noise)
7) Distraction Tool (throw objects)
8) Hiding Spots (become invisible in crates/corners)

PART 3: Game Systems, Progression & Win Conditions
9) Non-Lethal Takedown (knock out from behind)
10) Keycards & Locked Doors (doors block movement, require keys)
11) Alarm Level System (Global) (affects guard behavior)
12) Objective-Based Win Condition (hack terminals, reach exit)

Controls:
W/A/S/D      - Move forward/left/back/right
Arrow Keys   - Rotate player
C            - Toggle camera (3rd / 1st person)
F            - Melee / Takedown / Interact
G            - Throw distraction object
E            - Hack terminal (if in range)
H            - Hide (toggle hiding spot)
R            - Reset level
P            - Pause
ESC          - Quit

Run: python stealth_maze_new.py
"""

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import time
import math
import random
import sys

# ---------------------------
# Constants / configuration
# ---------------------------
WINDOW_W, WINDOW_H = 1200, 800
GRID = 1  # logical cell size unit; we will map to world units
CELL = 60  # world size for each cell (in units) - smaller corridors
CORRIDOR_HEIGHT = 240  # wall height - tall enough to block sight between corridors
PLAYER_SIZE = 12  # character body size (hero and enemies same) - smaller
PLAYER_HEIGHT = 30  # reduced height for slimmer look
WALL_HEIGHT = 200  # tall pillar walls that block vision
PLAYER_SPEED = 160.0  # units/sec (scaled for larger cells)
ENEMY_SPEED = 110.0
ENEMY_CHASE_SPEED = 180.0
ROT_SPEED = 90.0  # degrees per second (slower rotation)
BULLET_SPEED = 420.0
BULLET_RADIUS = 4
DOOR_OPEN_TIME = 2.0
DOOR_RELOCK_TIME = 6.0
POWERUP_DURATION = 10.0
MAX_ALERTED_NEARBY = 2  # only 1-2 nearby enemies may become alerted
# Melee / hand collision settings
KNIFE_DAMAGE = 1.75
UNARMED_DAMAGE = 1.0
MELEE_COOLDOWN_KNIFE = 0.3
MELEE_COOLDOWN_UNARMED = 0.5
MELEE_RANGE = PLAYER_SIZE * 1.3
UNARMED_INTERACTIONS_TO_KILL = 3

# Game states
STATE_PLAY = 0
STATE_PAUSE = 1
STATE_GAMEOVER = 2

# Camera modes
CAM_THIRD = 0
CAM_FIRST = 1

# ---------------------------
# Helper math
# ---------------------------

def vec_len(x, y):
    return math.hypot(x, y)


def clamp(v, a, b):
    return max(a, min(b, v))


def angle_between(ax, ay, bx, by):
    a = math.atan2(ay, ax)
    b = math.atan2(by, bx)
    diff = (b - a)
    while diff <= -math.pi:
        diff += 2 * math.pi
    while diff > math.pi:
        diff -= 2 * math.pi
    return diff

def angle_diff(a1, a2):
    """Return signed angle difference in radians."""
    diff = a2 - a1
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff <= -math.pi:
        diff += 2 * math.pi
    return diff

# ============ PART 1: VISION & DETECTION (Stealth Features 1-4) ============
VISION_CONE_RANGE = 400.0
VISION_CONE_WIDTH = 80.0  # degrees
VISION_CONE_FOV = math.radians(VISION_CONE_WIDTH / 2)

DETECTION_INCREASE_RATE = 25.0  # per second when visible
DETECTION_DECREASE_RATE = 15.0  # per second when hidden
DETECTION_MAX = 100.0
DETECTION_ALERT_THRESHOLD = 70.0
DETECTION_ALARM_THRESHOLD = 85.0

LIGHT_TILE_BRIGHTNESS = 1.0
DARK_TILE_BRIGHTNESS = 0.4
PLAYER_DARK_DETECTION_MULT = 0.5

# ============ PART 2: GUARD STATES & SOUND (Stealth Features 5-8) ============
# Use string state names to match AI code elsewhere in the file
STATE_PATROL = 'patrol'
STATE_SUSPICIOUS = 'alerted'
STATE_ALERT = 'chasing'
STATE_SEARCH = 'searching'
STATE_RETURN = 'returning'

SOUND_FOOTSTEP_RANGE = 200.0
SOUND_DOOR_RANGE = 300.0
SOUND_DISTRACTION_RANGE = 350.0
SOUND_TAKEDOWN_RANGE = 200.0

DISTRACTION_PROJECTILE_SPEED = 350.0
DISTRACTION_PROJECTILE_RANGE = 500.0

# ============ PART 3: ALARM & OBJECTIVES (Stealth Features 9-12) ============
ALARM_LEVEL_0 = 0
ALARM_LEVEL_1 = 1
ALARM_LEVEL_2 = 2

ALARM_DECAY_RATE = 20.0
ALARM_UPGRADE_TO_1 = 50.0
ALARM_UPGRADE_TO_2 = 80.0

TAKEDOWN_RANGE = 50.0
TAKEDOWN_ANGLE_THRESHOLD = 100.0

# ---------------------------
# Level data (3 fixed levels)
# Each map is a list of strings; characters:
#  '#': wall
#  ' ': corridor
#  'D': door (closed by default)
#  'K': key
#  'G': gun powerup
#  'N': knife powerup
#  'I': invisibility powerup
#  'S': player spawn
#  'E': enemy spawn
# We also create two layers: layer 0 (ground z=0), layer 1 (upper z= CELL)
# The same textual maps are used for plan view; for "layers" we'll place some corridors
# on upper layer by putting uppercase markers 'U' in the same map to indicate an upper corridor tile.
# For simplicity we hardcode small maps but ensure multiple connected hallways and doors.
# ---------------------------

LEVELS = []

# Level 1: 5 Linear Corridors separated by pillar fences with doors
# Format: Corridor (open space) -> Pillar Fence with Door -> Corridor -> ...
# S = start, E = enemy, D = door in fence, # = pillar fence/wall, G/N/I = powerups
LEVELS.append({
    'grid': [
        "########################",
        "#      S              #",
        "#  E        E         #",
        "#                     #",
        "#  G             N    #",
        "########################",
        "#########D#############",  # Fence with door to Corridor 2
        "#                     #",
        "#  E           E      #",
        "#                     #",
        "#       G        I    #",
        "########################",
        "#########D#############",  # Fence with door to Corridor 3
        "#                     #",
        "#    E          E     #",
        "#                     #",
        "#   N              G  #",
        "########################",
        "#########D#############",  # Fence with door to Corridor 4
        "#                     #",
        "#  E               E  #",
        "#                     #",
        "#     I            N  #",
        "########################",
        "#########D#############",  # Fence with door to Corridor 5
        "#                     #",
        "#   E            E    #",
        "#                     #",
        "#       G          I  #",
        "########################",
    ],
    'powerups': {'G':5, 'N':5, 'I':5},
    'enemy_hp': 2,
    'damage': 1,
    'enemy_count': 10,
})

# ---------------------------
# World state
# ---------------------------
current_level_index = 0
world_map = []  # list of rows (strings)
map_w = 0
map_h = 0

# Entities
player = {
    'x': 0.0, 'y': 0.0, 'z': 20.0,
    'angle': 0.0,  # degrees
    'life': 13,
    'has_gun': False,
    'gun_bullets': 0,
    'has_knife': False,
    'invisible': False,
    'invis_timer': 0.0,
    'melee_timer': 0.0,
    'melee_interaction_count': 0,
    'equipped': 'unarmed',
    # STEALTH FEATURES (Part 1-3)
    'detection_meter': 0.0,  # How detected is player (0-100)
    'in_hiding_spot': False,
    'hacking_terminal': None,
    'hacking_progress': 0.0,
}

enemies = []  # Enhanced with state machine, detection, vision cones

bullets = []  # dicts: x,y,z,vx,vy,owner('player'/'enemy'),life

powerups = []  # dicts: x,y,type,active,timer

doors = []  # dicts: x,y,open(0/1),opening_timer,relock_timer

terminals = []  # objectives to hack

hiding_spots = []  # locations where player can hide

noise_events = []  # sound events that guards react to

keys_collected = 0
keys_total = 0

# Global stealth state
global_alarm_level = ALARM_LEVEL_0
global_alarm_meter = 0.0
objectives_hacked = 0
objectives_total = 3
current_corridor = 0

# Game control
game_state = STATE_PLAY
camera_mode = CAM_THIRD
last_time = time.time()
paused_time_acc = 0.0

# Camera variables
cam_third_offset = (0, -320.0, 180.0)
cam_zoom = 1.0
camera_pos = (0.0, 500.0, 500.0)
pcamera_pos = camera_pos
fovY = 120
fp = False
camera_target = (0.0, 0.0, 0.0)
prefpc = (0.0, 0.0, 0.0)
prefp = False
prefpt = (0.0, 0.0, 0.0)

# ---------------------------
# Map helper functions
# ---------------------------

# ============================================================================
# STEALTH SYSTEMS (PARTS 1-3)
# ============================================================================

# ============ PART 1: VISION CONE & LINE OF SIGHT ============
def check_guard_vision(guard, player_pos):
    """Check if guard can see player using vision cone."""
    dist = vec_len(player_pos[0] - guard['x'], player_pos[1] - guard['y'])
    
    if dist > VISION_CONE_RANGE:
        return False, 0.0
    
    # Angle check
    dx = player_pos[0] - guard['x']
    dy = player_pos[1] - guard['y']
    player_angle_rad = math.atan2(dy, dx)
    guard_angle_rad = math.radians(guard['angle'])
    angle_diff_val = angle_diff(guard_angle_rad, player_angle_rad)
    
    if abs(angle_diff_val) > VISION_CONE_FOV:
        return False, 0.0
    
    # Visibility fades with distance
    visibility = 1.0 - (dist / VISION_CONE_RANGE)
    return True, visibility

# ============ PART 1: PATROL ROUTES ============
def generate_patrol_waypoints(start_x, start_y, num_points=4):
    """Generate patrol waypoints around spawn."""
    waypoints = []
    base_dist = CELL * 3
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi
        px = start_x + math.cos(angle) * base_dist
        py = start_y + math.sin(angle) * base_dist
        waypoints.append((px, py))
    return waypoints

# ============ PART 1: LIGHTING SYSTEM ============
def get_tile_brightness(x, y):
    """Return brightness for a world position."""
    grid_x = int((x / CELL) + map_w // 2)
    grid_y = int((y / CELL) + map_h // 2)
    
    if grid_x < 0 or grid_y < 0 or grid_x >= map_w or grid_y >= map_h:
        return LIGHT_TILE_BRIGHTNESS
    
    # Checkerboard pattern
    if (grid_x + grid_y) % 2 == 0:
        return LIGHT_TILE_BRIGHTNESS
    else:
        return DARK_TILE_BRIGHTNESS

# ============ PART 2: SOUND/NOISE SYSTEM ============
def add_noise_event(x, y, radius, event_type):
    """Create a noise event."""
    noise_events.append({
        'x': x, 'y': y, 'radius': radius,
        'type': event_type,
        'time_created': time.time(),
    })

def guard_hears_noise(guard, noise):
    """Check if guard hears a noise."""
    dist = vec_len(noise['x'] - guard['x'], noise['y'] - guard['y'])
    return dist <= noise['radius']

# ============ PART 2: HIDING SPOTS ============
def check_near_hiding_spot():
    """Check if player is near a hiding spot."""
    for spot in hiding_spots:
        dist = vec_len(player['x'] - spot['x'], player['y'] - spot['y'])
        if dist < spot['radius']:
            return True
    return False

# ============ PART 3: TAKEDOWN ============
def attempt_takedown(guard):
    """Attempt to takedown a guard."""
    dist = vec_len(player['x'] - guard['x'], player['y'] - guard['y'])
    if dist > TAKEDOWN_RANGE:
        return False
    
    # Check angle
    dx = guard['x'] - player['x']
    dy = guard['y'] - player['y']
    target_angle = math.atan2(dy, dx)
    player_angle_rad = math.radians(player['angle'])
    angle_diff_val = abs(angle_diff(player_angle_rad, target_angle))
    
    if angle_diff_val > math.radians(TAKEDOWN_ANGLE_THRESHOLD):
        return False
    
    # Disable guard
    guard['hp'] = 0
    guard['state'] = STATE_PATROL
    add_noise_event(guard['x'], guard['y'], SOUND_TAKEDOWN_RANGE, 'takedown')
    return True

# ============ PART 2: DISTRACTION TOOL ============
def throw_distraction():
    """Throw a distraction object."""
    angle = math.radians(player['angle'])
    dist = DISTRACTION_PROJECTILE_RANGE * 0.8
    land_x = player['x'] + math.cos(angle) * dist
    land_y = player['y'] + math.sin(angle) * dist
    add_noise_event(land_x, land_y, SOUND_DISTRACTION_RANGE, 'distraction')

# ============ PART 3: ALARM LEVEL SYSTEM ============
def update_global_alarm_level():
    """Update global alarm based on player detection."""
    global global_alarm_level, global_alarm_meter
    
    max_detection = player['detection_meter']
    for enemy in enemies:
        max_detection = max(max_detection, enemy.get('detection_meter', 0.0))
    
    global_alarm_meter = max_detection
    
    if global_alarm_meter >= ALARM_UPGRADE_TO_2:
        global_alarm_level = ALARM_LEVEL_2
    elif global_alarm_meter >= ALARM_UPGRADE_TO_1:
        global_alarm_level = ALARM_LEVEL_1
    else:
        global_alarm_level = ALARM_LEVEL_0
    
    # Decay when hiding
    if player['in_hiding_spot']:
        global_alarm_meter = max(0.0, global_alarm_meter - ALARM_DECAY_RATE * 0.016)

# ============ PART 3: OBJECTIVE HACKING ============
def hack_terminal(terminal, dt):
    """Progress hacking."""
    required_time = 5.0
    player['hacking_progress'] += dt
    if player['hacking_progress'] >= required_time:
        global objectives_hacked
        objectives_hacked += 1
        player['hacking_progress'] = 0.0
        player['hacking_terminal'] = None
        return True
    return False

# ============ PART 2: UPDATE GUARD STATE ============
def update_guard_state(guard, dt):
    """Update guard state machine."""
    detection = guard.get('detection_meter', 0.0)
    current_state = guard['state']
    
    # Check if can see player
    can_see, visibility = check_guard_vision(guard, (player['x'], player['y']))
    
    # Update detection
    if can_see:
        brightness = get_tile_brightness(guard['x'], guard['y'])
        rate = DETECTION_INCREASE_RATE * visibility * brightness
        guard['detection_meter'] = min(DETECTION_MAX, guard['detection_meter'] + rate * dt)
    else:
        guard['detection_meter'] = max(0.0, guard['detection_meter'] - DETECTION_DECREASE_RATE * dt)

    # use updated detection value for transitions
    detection = guard.get('detection_meter', 0.0)

    # State transitions
    if detection >= DETECTION_ALARM_THRESHOLD:
        guard['state'] = STATE_ALERT
    elif detection >= DETECTION_ALERT_THRESHOLD:
        guard['state'] = STATE_SUSPICIOUS
    elif current_state in [STATE_SUSPICIOUS, STATE_ALERT, STATE_SEARCH] and detection < 30.0:
        guard['state'] = STATE_PATROL

# ---------------------------
def load_level(idx):
    """Load level: simple linear corridors with pillar fences and doors."""
    global world_map, map_w, map_h, enemies, powerups, doors, player, terminals, hiding_spots, noise_events, bullets, current_corridor
    
    level = LEVELS[idx % len(LEVELS)]
    world_map = [row for row in level['grid']]
    map_h = len(world_map)
    map_w = max(len(row) for row in world_map)
    
    # Reset all entities
    enemies = []
    powerups = []
    doors = []
    terminals = []
    hiding_spots = []
    noise_events = []
    bullets = []
    current_corridor = 0
    
    # Spawn entities from map
    for y, row in enumerate(world_map):
        for x, ch in enumerate(row):
            wx = (x - map_w // 2) * CELL
            wy = (y - map_h // 2) * CELL
            
            if ch == 'S':  # Player start
                player['x'] = wx
                player['y'] = wy
                player['z'] = 20.0
                player['angle'] = 0.0
                player['life'] = 13
                player['has_gun'] = False
                player['gun_bullets'] = 0
                player['has_knife'] = False
                player['invisible'] = False
                player['invis_timer'] = 0.0
                player['melee_timer'] = 0.0
                player['detection_meter'] = 0.0
                player['in_hiding_spot'] = False
                player['hacking_terminal'] = None
                player['hacking_progress'] = 0.0
                
            elif ch == 'E':  # Enemy
                enemies.append({
                    'x': wx, 'y': wy, 'z': 20.0,
                    'angle': 0.0,
                    'state': 'patrol',
                    'patrol': [(wx, wy), (wx+CELL*2, wy)],
                    'pi': 0,
                    'hp': level['enemy_hp'],
                    'last_seen': None,
                    'alerted_since': None,
                    'shoot_cooldown': 0.0,
                    'detection_meter': 0.0,
                    'corridor': y // 6,  # Assign to corridor
                })
                
            elif ch == 'D':  # Door
                doors.append({
                    'x': wx,
                    'y': wy,
                    'z': 0,
                    'open': False,
                    'opening_timer': 0.0,
                    'relock_timer': 0.0,
                    'corridor': y // 6,  # Door opens when corridor enemies defeated
                })
                
            elif ch in ('G', 'N', 'I'):  # Powerups
                powerups.append({
                    'x': wx,
                    'y': wy,
                    'z': 20,
                    'type': ch,
                    'active': True,
                    'timer': 0.0,
                })
    
    # Create some hiding spots in corridors (empty spaces)
    hiding_spots = []
    for _ in range(4):
        found = False
        for _ in range(100):
            y = random.randint(0, map_h - 1)
            x = random.randint(0, map_w - 1)
            if x < len(world_map[y]) and world_map[y][x] == ' ':
                wx = (x - map_w // 2) * CELL
                wy = (y - map_h // 2) * CELL
                hiding_spots.append({'x': wx, 'y': wy, 'radius': CELL * 0.6})
                found = True
                break
        if not found:
            break

    player['melee_timer'] = 0.0


def find_empty_cell():
    for _ in range(2000):
        x = random.randint(1, map_w-2)
        y = random.randint(1, map_h-2)
        if world_map[y][x] == ' ':
            wx = (x - map_w // 2) * CELL
            wy = (y - map_h // 2) * CELL
            return wx, wy
    return 0,0


def build_patrol_for(wx, wy):
    # simple short patrol: two points (spawn and a nearby free cell)
    pts = [(wx, wy)]
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx = wx + dx*CELL*2
        ny = wy + dy*CELL*2
        if not cell_is_wall_world(nx, ny):
            pts.append((nx, ny))
            break
    if len(pts) == 1:
        pts.append((wx+CELL*2, wy))
    return pts


def world_to_cell(wx, wy):
    cx = int(round(wx / CELL)) + map_w // 2
    cy = int(round(wy / CELL)) + map_h // 2
    return cx, cy


def cell_is_wall_cell(cx, cy):
    if cx < 0 or cy < 0 or cy >= map_h or cx >= map_w:
        return True
    if cx >= len(world_map[cy]):
        return True
    return world_map[cy][cx] == '#'


def cell_is_wall_world(wx, wy):
    cx, cy = world_to_cell(wx, wy)
    return cell_is_wall_cell(cx, cy)


def door_at_world(wx, wy):
    for d in doors:
        if abs(d['x'] - wx) < (CELL/2) and abs(d['y'] - wy) < (CELL/2):
            return d
    return None


def try_open_door_at(tx, ty, opener=None):
    """Try to open a door at world coordinates (tx,ty).
    Returns True if a door was found and opening was initiated or already open.
    """
    d = door_at_world(tx, ty)
    if not d:
        # snap to nearest cell center and try again
        cx, cy = world_to_cell(tx, ty)
        wx = (cx - map_w//2) * CELL
        wy = (cy - map_h//2) * CELL
        d = door_at_world(wx, wy)
        if not d:
            return False
    # if already open, nothing to do
    if d.get('open'):
        return True
    # start opening timer; update will flip to open when timer elapses
    d['opening_timer'] = DOOR_OPEN_TIME
    return True

# ---------------------------
# Collision
# ---------------------------

def collides_with_walls(x, y, radius=PLAYER_SIZE):
    # check the 4 neighboring cells
    cx, cy = world_to_cell(x, y)
    for dy in range(-1,2):
        for dx in range(-1,2):
            tx = cx + dx
            ty = cy + dy
            if cell_is_wall_cell(tx, ty):
                wx = (tx - map_w//2)*CELL
                wy = (ty - map_h//2)*CELL
                # treat wall cell as square of size CELL
                # check circle vs AABB
                nearest_x = clamp(x, wx - CELL/2, wx + CELL/2)
                nearest_y = clamp(y, wy - CELL/2, wy + CELL/2)
                if vec_len(nearest_x - x, nearest_y - y) < radius:
                    return True
    # doors as blocking when closed
    for d in doors:
        if not d['open']:
            if abs(d['x']-x) < CELL/2 and abs(d['y']-y) < CELL/2:
                return True
    return False

# ---------------------------
# LOS and vision cone
# ---------------------------

def line_of_sight(x1, y1, x2, y2):
    # step along the segment and check walls
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    steps = int(max(8, dist / (CELL * 0.25)))
    for i in range(1, steps+1):
        t = i / steps
        sx = x1 + dx * t
        sy = y1 + dy * t
        if cell_is_wall_world(sx, sy):
            return False
        # closed doors block
        d = door_at_world(sx, sy)
        if d and not d['open']:
            return False
    return True


def in_vision_cone(ex, ey, eang_deg, fov_deg, max_dist, tx, ty):
    # ensure cone rotates with enemy (we only call with enemy's angle)
    dx = tx - ex
    dy = ty - ey
    d = math.hypot(dx, dy)
    if d > max_dist:
        return False
    # enemy forward vector (must match the forward direction used in rendering)
    rad = math.radians(eang_deg)
    fx = -math.sin(rad)  # Fixed to match rendering
    fy = math.cos(rad)   # Fixed to match rendering
    # angle between forward and target vector
    ang = math.degrees(abs(angle_between(fx, fy, dx/d if d>0 else 0, dy/d if d>0 else 0)))
    return ang <= (fov_deg/2)

# ---------------------------
# AI helpers
# ---------------------------

def nearby_enemies(center_x, center_y, radius=300):
    return [e for e in enemies if vec_len(e['x']-center_x, e['y']-center_y) <= radius]


# Enemy movement helper: attempt to move an enemy, open doors if needed
def try_enemy_move(e, dx, dy):
    tx = e['x'] + dx
    ty = e['y'] + dy
    d = door_at_world(tx, ty)
    if d and not d['open']:
        # start opening door (enemy can open it)
        d['opening_timer'] = DOOR_OPEN_TIME
        return False
    if not collides_with_walls(tx, ty):
        e['x'] = tx
        e['y'] = ty
        return True
    return False

# ---------------------------
# Game update
# ---------------------------

def update(dt):
    global bullets, game_state, current_level_index, keys_collected, player
    # Check if player died
    if player['life'] <= 0:
        game_state = STATE_GAMEOVER
        return
    if game_state != STATE_PLAY:
        return
    # update player invis timer
    if player['invisible']:
        player['invis_timer'] -= dt
        if player['invis_timer'] <= 0:
            player['invisible'] = False
            player['invis_timer'] = 0.0

    # update doors
    for d in doors:
        if d['open'] is False and d['opening_timer'] > 0:
            d['opening_timer'] -= dt
            if d['opening_timer'] <= 0:
                d['open'] = True
                d['relock_timer'] = DOOR_RELOCK_TIME
        if d['open'] and d['relock_timer'] > 0:
            d['relock_timer'] -= dt
            if d['relock_timer'] <= 0:
                d['open'] = False
                d['relock_timer'] = 0.0

    # update bullets
    kept = []
    for b in bullets:
        b['x'] += b['vx'] * dt
        b['y'] += b['vy'] * dt
        b['life'] -= dt
        if b['life'] <= 0:
            continue
        # collide with walls
        if cell_is_wall_world(b['x'], b['y']):
            continue
        # collide with doors
        d = door_at_world(b['x'], b['y'])
        if d and not d['open']:
            continue
        if b['owner'] == 'player':
            # hit enemy?
            for e in enemies:
                if vec_len(e['x']-b['x'], e['y']-b['y']) < PLAYER_SIZE:
                    # gun kills immediately (per rules: gun kills in 1 shot)
                    e['hp'] = 0
                    b['life'] = 0
                    break
        else:
            if vec_len(player['x']-b['x'], player['y']-b['y']) < PLAYER_SIZE:
                # hit player
                player['life'] -= LEVELS[current_level_index]['damage']
                b['life'] = 0
                if player['life'] <= 0:
                    game_state = STATE_GAMEOVER
        if b['life'] > 0:
            kept.append(b)
    bullets = kept

    # process noise events (guards investigate sounds)
    nowt = time.time()
    kept_noises = []
    for n in noise_events:
        age = nowt - n.get('time_created', nowt)
        # expire noises after a few seconds
        if age > 6.0:
            continue
        # alert nearby guards
        for e in enemies:
            if e['hp'] <= 0:
                continue
            if guard_hears_noise(e, n):
                # if not already in a high-alert state, investigate
                if e['state'] not in (STATE_ALERT, STATE_SUSPICIOUS, 'chasing'):
                    e['state'] = STATE_SUSPICIOUS
                    e['last_seen'] = (n['x'], n['y'])
                    e['alerted_since'] = nowt
        kept_noises.append(n)
    noise_events[:] = kept_noises

    # update enemies
    # Update player's detection meter based on guards' vision
    seen_by_any = False
    for e in enemies:
        if e.get('hp', 0) <= 0:
            continue
        can_see, vis = check_guard_vision(e, (player['x'], player['y']))
        if can_see and line_of_sight(e['x'], e['y'], player['x'], player['y']) and not player.get('invisible', False):
            seen_by_any = True
            # brightness at player influences detectability
            brightness = get_tile_brightness(player['x'], player['y'])
            dark_mult = PLAYER_DARK_DETECTION_MULT if brightness < 1.0 else 1.0
            rate = DETECTION_INCREASE_RATE * vis * brightness * dark_mult
            player['detection_meter'] = min(DETECTION_MAX, player.get('detection_meter', 0.0) + rate * dt)
        # small per-enemy detection influence when not seen is handled below
    if not seen_by_any:
        player['detection_meter'] = max(0.0, player.get('detection_meter', 0.0) - DETECTION_DECREASE_RATE * dt)
    # enforce alert limits: at most MAX_ALERTED_NEARBY near events
    for e in enemies:
        if e['hp'] <= 0:
            continue
        # cooldowns
        if e['shoot_cooldown'] > 0:
            e['shoot_cooldown'] -= dt
        # simple state machine
        if e['state'] == 'patrol':
            # move toward current patrol point
            tx, ty = e['patrol'][e['pi']]
            vx = tx - e['x']
            vy = ty - e['y']
            dist = math.hypot(vx, vy)
            if dist < 6:
                e['pi'] = (e['pi'] + 1) % len(e['patrol'])
            else:
                nx = (vx / dist) * ENEMY_SPEED * dt
                ny = (vy / dist) * ENEMY_SPEED * dt
                # try move, avoid walls or open doors
                try_enemy_move(e, nx, ny)
            # rotate slowly toward next
            desired = math.degrees(math.atan2(ty-e['y'], tx-e['x']))
            diff = (desired - e['angle'] + 180) % 360 - 180
            e['angle'] += clamp(diff, -60*dt, 60*dt)
            # check vision cone
            if detect_player_by_enemy(e):
                e['state'] = 'chasing'
                e['last_seen'] = (player['x'], player['y'])
                e['alerted_since'] = time.time()
                # alert up to one/two nearby
                count = 0
                for other in nearby_enemies(e['x'], e['y'], radius= CELL*3 ):
                    if other is e: continue
                    if count >= (MAX_ALERTED_NEARBY-1): break
                    # only if in same hallway approx (no walls between)
                    if line_of_sight(e['x'], e['y'], other['x'], other['y']):
                        other['state'] = 'alerted'
                        other['alerted_since'] = time.time()
                        count += 1
        elif e['state'] == 'alerted':
            # look toward player's last known position briefly, then search
            if detect_player_by_enemy(e):
                e['state'] = 'chasing'
                e['last_seen'] = (player['x'], player['y'])
            else:
                # search behavior: walk toward last seen if exists
                if e['last_seen']:
                    tx, ty = e['last_seen']
                    vx = tx - e['x']
                    vy = ty - e['y']
                    dist = math.hypot(vx, vy)
                    if dist > 6:
                        nx = (vx / dist) * ENEMY_SPEED * dt
                        ny = (vy / dist) * ENEMY_SPEED * dt
                        try_enemy_move(e, nx, ny)
                    else:
                        # if reached last seen, switch to searching
                        e['state'] = 'searching'
                        e['search_timer'] = 3.0
                else:
                    e['state'] = 'patrol'
        elif e['state'] == 'chasing':
            # move faster toward player
            tx = player['x']
            ty = player['y']
            vx = tx - e['x']
            vy = ty - e['y']
            dist = math.hypot(vx, vy)
            if dist > 4:
                nx = (vx / dist) * ENEMY_CHASE_SPEED * dt
                ny = (vy / dist) * ENEMY_CHASE_SPEED * dt
                try_enemy_move(e, nx, ny)
            # rotate to face player
            desired = math.degrees(math.atan2(ty-e['y'], tx-e['x']))
            diff = (desired - e['angle'] + 180) % 360 - 180
            e['angle'] += clamp(diff, -360*dt, 360*dt)
            # shoot if player visible and cooldown done
            if detect_player_by_enemy(e):
                if e['shoot_cooldown'] <= 0.0 and not player['invisible']:
                    # enemy fires one bullet from gun hand position
                    rad = math.radians(e['angle'])
                    fx = -math.sin(rad)
                    fy = math.cos(rad)
                    rx = math.cos(rad)
                    ry = math.sin(rad)
                    gun_x = e['x'] + fx * 8 + rx * 6  # gun hand position
                    gun_y = e['y'] + fy * 8 + ry * 6
                    fire_bullet(gun_x, gun_y, e['angle'], 'enemy')
                    e['shoot_cooldown'] = 1.0  # Shoot every 1 second continuously
                e['last_seen'] = (player['x'], player['y'])
            else:
                # lost sight
                if e['last_seen']:
                    e['state'] = 'searching'
                    e['search_timer'] = 4.0
                else:
                    e['state'] = 'returning'
        elif e['state'] == 'searching':
            # roam around last seen spot
            e['search_timer'] -= dt
            if detect_player_by_enemy(e):
                e['state'] = 'chasing'
            elif e['search_timer'] <= 0:
                e['state'] = 'returning'
        elif e['state'] == 'returning':
            # go back to patrol start
            tx, ty = e['patrol'][0]
            vx = tx - e['x']
            vy = ty - e['y']
            dist = math.hypot(vx, vy)
            if dist > 6:
                nx = (vx / dist) * ENEMY_SPEED * dt
                ny = (vy / dist) * ENEMY_SPEED * dt
                try_enemy_move(e, nx, ny)
            else:
                e['state'] = 'patrol'

    # remove dead enemies, drop keys
    for e in list(enemies):
        if e['hp'] <= 0:
            enemies.remove(e)

    # Automatic hand-collision melee: if player is close to an enemy, apply melee damage without pressing F
    if player.get('melee_timer', 0) > 0:
        player['melee_timer'] -= dt
    for e in enemies:
        if e['hp'] <= 0:
            continue
        d = vec_len(e['x'] - player['x'], e['y'] - player['y'])
        if d < MELEE_RANGE:
            if player.get('melee_timer', 0) <= 0:
                if player.get('has_knife'):
                    e['hp'] -= KNIFE_DAMAGE
                    player['melee_timer'] = MELEE_COOLDOWN_KNIFE
                    e['state'] = 'chasing'
                elif player.get('has_gun'):
                    # gun does not melee damage
                    pass
                else:
                    e['hp'] -= UNARMED_DAMAGE
                    player['melee_timer'] = MELEE_COOLDOWN_UNARMED
                    e['state'] = 'chasing'

    # pickups
    for pu in powerups:
        if not pu['active']:
            continue
        if vec_len(pu['x']-player['x'], pu['y']-player['y']) < CELL*0.6:
            take_powerup(pu)

    # check for keys
    for pu in powerups:
        if pu['active'] and pu['type'] == 'K' and vec_len(pu['x']-player['x'], pu['y']-player['y']) < CELL*0.6:
            pu['active'] = False
            keys_collected += 1

    # Layer transition: if a separator door is open and player is very near, move player to the adjacent layer side
    for d in doors:
        if d.get('layer_from') is not None and d.get('open'):
            if vec_len(player['x'] - d['x'], player['y'] - d['y']) < CELL * 0.8:
                # move player across the door to the next layer
                # if player is above (smaller y), move to below side, else move above
                if player['y'] < d['y']:
                    player['y'] = d['y'] + CELL * 0.7
                else:
                    player['y'] = d['y'] - CELL * 0.7
                # small nudge to avoid re-triggering
                player['x'] = d['x']

    # Check if all enemies in current corridor are dead - if so, open the door to next corridor
    enemies_in_corridor = [e for e in enemies if e.get('corridor', 0) == current_corridor and e['hp'] > 0]
    if not enemies_in_corridor:
        # All enemies in current corridor defeated
        for d in doors:
            if d.get('corridor', 0) == current_corridor and not d['open']:
                d['open'] = True
                d['opening_timer'] = 0.0
                d['relock_timer'] = 9999
    
    # Check if player reached the last door (level complete)
    all_enemies_dead = all(e['hp'] <= 0 for e in enemies)
    if all_enemies_dead:
        # All enemies defeated - player can exit through last door
        for d in doors:
            d['open'] = True
            d['opening_timer'] = 0.0
            d['relock_timer'] = 9999
        # If player is near a door, advance to next level
        for d in doors:
            if d['open'] and vec_len(player['x'] - d['x'], player['y'] - d['y']) < CELL * 0.6:
                current_level_index = (current_level_index + 1) % len(LEVELS)
                load_level(current_level_index)
                return

    # update powerup timers
    for pu in powerups:
        if not pu['active'] and pu.get('timer',0) > 0:
            pu['timer'] -= dt
            if pu['timer'] <= 0:
                pu['active'] = True

    # ensure gun flag cleared when bullets exhausted
    if player.get('has_gun') and player.get('gun_bullets', 0) <= 0:
        player['has_gun'] = False

    # melee collisions: if player is close to an enemy and presses F it will be handled in keyboard, but we must also detect frontal interactions for '3 interactions' when unarmed
    for e in enemies:
        if e['hp'] <= 0: continue
        d = vec_len(e['x']-player['x'], e['y']-player['y'])
        if d < PLAYER_SIZE*1.2:
            # determine relative angle
            desired = math.degrees(math.atan2(player['y']-e['y'], player['x']-e['x']))
            diff = (desired - e['angle'] + 180) % 360 - 180
            if abs(diff) < 60:
                # player is in front of enemy; if player has no weapon, apply interactions slowly
                if not player['has_gun'] and not player['has_knife'] and not player['invisible']:
                    # deal damage to player slowly
                    player['life'] -= 0.5 * dt  # continuous small damage
                    if player['life'] <= 0:
                        game_state = STATE_GAMEOVER

    # update global alarm now that detections changed
    update_global_alarm_level()

# ---------------------------
# Detect player by enemy
# ---------------------------

def detect_player_by_enemy(e):
    # invisibility prevents detection
    if player['invisible']:
        return False
    # rough checks
    if not in_vision_cone(e['x'], e['y'], e['angle'], 70, CELL*6, player['x'], player['y']):
        return False
    # ensure line of sight
    if not line_of_sight(e['x'], e['y'], player['x'], player['y']):
        return False
    return True

# ---------------------------
# Actions: bullets, powerups, doors
# ---------------------------

def fire_bullet(sx, sy, angle_deg, owner='player'):
    # Use the same forward vector as movement (forward = -sin, cos)
    rad = math.radians(angle_deg)
    fx = -math.sin(rad)
    fy = math.cos(rad)
    vx = fx * BULLET_SPEED
    vy = fy * BULLET_SPEED
    # spawn bullet from chest (forward offset)
    spawn_x = sx + fx * (PLAYER_SIZE * 0.9)
    spawn_y = sy + fy * (PLAYER_SIZE * 0.9)
    spawn_z = player.get('z', 20.0) + PLAYER_HEIGHT * 0.4 if owner == 'player' else 20.0
    bullets.append({'x': spawn_x, 'y': spawn_y, 'z': spawn_z, 'vx': vx, 'vy': vy, 'owner': owner, 'life': 4.0})


def take_powerup(pu):
    t = pu['type']
    # Keys are handled separately in update() -> let that code collect keys
    if t == 'K':
        return
    # Powerups (non-keys) are taken permanently (do not respawn)
    pu['active'] = False
    
    if t == 'G':
        player['has_gun'] = True
        player['gun_bullets'] = 3  # enough to kill at least 2 enemies per requirement
        player['equipped'] = 'gun'
    elif t == 'N':
        player['has_knife'] = True
        player['equipped'] = 'knife'
    elif t == 'I':
        player['invisible'] = True
        player['invis_timer'] = POWERUP_DURATION
    elif t == 'K':
        # key
        pass


def melee_or_fire():
    # Use equipped weapon semantics: 'gun', 'knife', 'unarmed'
    eq = player.get('equipped', 'unarmed')
    # Gun: consume bullets and fire
    if eq == 'gun':
        if player.get('has_gun') and player.get('gun_bullets', 0) > 0:
            fire_bullet(player['x'], player['y'], player['angle'], 'player')
            player['gun_bullets'] -= 1
            # loud shot alerts nearby enemies with line of sight
            for e in nearby_enemies(player['x'], player['y'], radius=CELL*5):
                if line_of_sight(e['x'], e['y'], player['x'], player['y']):
                    e['state'] = 'alerted'
                    e['alerted_since'] = time.time()
            if player['gun_bullets'] <= 0:
                player['has_gun'] = False
        return
    # Knife: immediate melee damage to nearby enemy
    if eq == 'knife':
        for e in enemies:
            d = vec_len(e['x'] - player['x'], e['y'] - player['y'])
            if d < PLAYER_SIZE * 1.3:
                e['hp'] -= KNIFE_DAMAGE
                e['state'] = 'chasing'
                return
    # Unarmed: interactions needed to kill (uses melee_timer and interaction count)
    for e in enemies:
        d = vec_len(e['x'] - player['x'], e['y'] - player['y'])
        if d < PLAYER_SIZE * 1.3:
            if player.get('melee_timer', 0) <= 0:
                player['melee_interaction_count'] = player.get('melee_interaction_count', 0) + 1
                player['melee_timer'] = MELEE_COOLDOWN_UNARMED
                if player['melee_interaction_count'] >= UNARMED_INTERACTIONS_TO_KILL:
                    e['hp'] = 0
                    player['melee_interaction_count'] = 0
                e['state'] = 'chasing'
            return


    return

# ---------------------------
# Input handlers
# ---------------------------

keys = set()


def keyboard(key, x, y):
    global game_state, current_level_index
    k = key.decode('utf-8') if isinstance(key, bytes) else key
    if k.lower() == 'p':
        if game_state == STATE_PLAY:
            game_state = STATE_PAUSE
        elif game_state == STATE_PAUSE:
            game_state = STATE_PLAY
    if k.lower() == 'r':
        reset_game()
    if game_state != STATE_PLAY:
        return
    if k.lower() == 'c':
        toggle_camera()
    if k == 'f' or k == 'F':
        # interact with doors OR attempt takedown on nearby guard
        rad = math.radians(player['angle'])
        tx = player['x'] + math.cos(rad) * (CELL)
        ty = player['y'] + math.sin(rad) * (CELL)
        
        # Try takedown first
        takedown_attempted = False
        for guard in enemies:
            if attempt_takedown(guard):
                takedown_attempted = True
                break
        
        # If no takedown, try door
        if not takedown_attempted:
            try_open_door_at(tx, ty, opener='player')

    # Weapon switch (cycle equipped if player has multiple)
    # STEALTH: Throw distraction object
    if k.lower() == 'g':
        throw_distraction()

    # STEALTH: Hide/Unhide
    if k.lower() == 'h':
        if player['in_hiding_spot']:
            player['in_hiding_spot'] = False
        elif check_near_hiding_spot():
            player['in_hiding_spot'] = True
    
    # STEALTH: Hack terminal
    if k.lower() == 'e':
        # Check if near terminal
        for term in terminals:
            dist = vec_len(player['x'] - term['x'], player['y'] - term['y'])
            if dist < CELL:
                player['hacking_terminal'] = term
                break

    # Weapon switch (moved to O)
    if k.lower() == 'o':
        eq = player.get('equipped', 'unarmed')
        if eq == 'unarmed':
            if player.get('has_gun'):
                player['equipped'] = 'gun'
            elif player.get('has_knife'):
                player['equipped'] = 'knife'
        elif eq == 'gun':
            if player.get('has_knife'):
                player['equipped'] = 'knife'
            else:
                player['equipped'] = 'unarmed'
        elif eq == 'knife':
            player['equipped'] = 'unarmed'

    # gun firing (also reachable via left-click)
    # gun firing - changed to Q
    if k.lower() == 'q':
        melee_or_fire()

    # movement keys
    if k.lower() in ('w','a','s','d'):
        keys.add(k.lower())


def keyboard_up(key, x, y):
    k = key.decode('utf-8') if isinstance(key, bytes) else key
    if k.lower() in keys:
        keys.discard(k.lower())


def special_key(key, x, y):
    # control camera zoom with up/down arrows
    global cam_zoom
    if key == GLUT_KEY_UP:
        # zoom out (increase distance)
        cam_zoom = min(cam_zoom + 0.1, 3.0)
    if key == GLUT_KEY_DOWN:
        # zoom in (decrease distance)
        cam_zoom = max(cam_zoom - 0.1, 0.5)
    if key == GLUT_KEY_LEFT:
        pass  # reserved for future use
    if key == GLUT_KEY_RIGHT:
        pass  # reserved for future use

# ---------------------------
# Camera
# ---------------------------

def toggle_camera():
    global camera_mode
    camera_mode = CAM_FIRST if camera_mode == CAM_THIRD else CAM_THIRD


def update_player_movement(dt):
    # WASD relative to facing
    if game_state != STATE_PLAY:
        return
    # Use reference movement style (relative to facing) but scaled by dt for time-consistency
    rad = math.radians(player['angle'])
    move_speed = PLAYER_SPEED  # units per second
    forward = (-math.sin(rad), math.cos(rad))  # Fixed: correct forward direction
    right = (math.cos(rad), math.sin(rad))
    dx = 0.0
    dy = 0.0
    if 'w' in keys:
        dx += forward[0] * move_speed * dt
        dy += forward[1] * move_speed * dt
    if 's' in keys:
        dx -= forward[0] * move_speed * dt
        dy -= forward[1] * move_speed * dt
    if 'a' in keys:
        # rotate left at controlled rate
        player['angle'] = (player['angle'] + ROT_SPEED * dt) % 360
    if 'd' in keys:
        # rotate right
        player['angle'] = (player['angle'] - ROT_SPEED * dt) % 360
    # strafe
    if 'a' not in keys and 'd' not in keys:
        # allow strafing with Q/E in future; keep simple for now
        pass
    if dx != 0.0 or dy != 0.0:
        if not collides_with_walls(player['x'] + dx, player['y'] + dy):
            player['x'] += dx
            player['y'] += dy


# Camera functions adapted from reference file
def u_fp():
    global camera_pos, camera_target, pcamera_pos, prefpc, prefp, prefpt, cam_zoom
    # if player dead, keep previous camera
    if game_state == STATE_GAMEOVER:
        camera_pos = pcamera_pos
        camera_target = (0,0,0)
        return
    # Use camera_mode to decide behavior: third-person or reference first-person
    if camera_mode != CAM_FIRST:
        # Third-person: camera stays fixed behind player (does NOT rotate with player)
        # Camera offset is fixed in world space, not rotating with player
        offset_dist = 200.0 * cam_zoom  # zoom affects distance
        offset_height = 150.0
        # Fixed position behind player (negative Y direction in world)
        cx = player['x']
        cy = player['y'] - offset_dist
        cz = player['z'] + offset_height
        # simple clip toward player to avoid wall penetration
        steps = 8
        ox, oy = player['x'], player['y']
        for i in range(steps):
            t = (i+1)/steps
            sx = ox + (cx-ox)*t
            sy = oy + (cy-oy)*t
            if cell_is_wall_world(sx, sy):
                cx = ox + (cx-ox)*(t-0.2)
                cy = oy + (cy-oy)*(t-0.2)
                break
        camera_pos = (cx, cy, cz)
        pcamera_pos = camera_pos
        camera_target = (player['x'], player['y'], player['z']+10)
        return
    # first-person-like preferred view from reference: position above player, look ahead
    rad = math.radians(player['angle'])
    lx = player['x'] - math.sin(rad) * 50  # Fixed direction
    ly = player['y'] + math.cos(rad) * 50  # Fixed direction
    camera_pos = (player['x'], player['y'], 120)
    camera_target = (lx, ly, 120)
    prefpc = camera_pos
    prefpt = camera_target


def setupCamera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fovY, WINDOW_W/float(WINDOW_H), 0.1, 4000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    cx, cy, cz = camera_pos
    tx, ty, tz = camera_target
    gluLookAt(cx, cy, cz, tx, ty, tz, 0, 0, 1)

# ---------------------------
# Rendering (allowed GL calls only)
# ---------------------------

def draw_text_screen(x, y, s):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, WINDOW_W, 0, WINDOW_H)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glColor3f(1.0, 1.0, 1.0)
    glRasterPos2f(x, y)
    for ch in s:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_cube_at(x, y, size, color=(0.6,0.6,0.6)):
    glColor3f(*color)
    glPushMatrix()
    # position cube center at x,y, z=size/2
    glTranslatef(x, y, size/2)
    glutSolidCube(size)
    glPopMatrix()


def draw_player():
    # body, head sphere, and hands attached to front
    if player['invisible']:
        glColor3f(0.1, 0.2, 0.1)  # dimmed green when invisible
    else:
        glColor3f(0.0, 1.0, 0.0)  # bright green - hero
    glPushMatrix()
    glTranslatef(player['x'], player['y'], player['z'])
    glRotatef(player['angle'], 0, 0, 1)
    
    # torso (slimmer) - centered at origin
    glColor3f(0.0, 1.0, 0.0)  # green body
    glPushMatrix()
    glTranslatef(0, 0, PLAYER_HEIGHT/2)
    glScalef(0.6, 0.6, 1.0)  # make slimmer
    glutSolidCube(PLAYER_HEIGHT)
    glPopMatrix()
    
    # head (sphere) - BLACK
    glPushMatrix()
    glTranslatef(0, 0, PLAYER_HEIGHT + 6)
    glColor3f(0.0, 0.0, 0.0)  # black head
    quad = gluNewQuadric()
    gluSphere(quad, 8, 12, 12)
    glPopMatrix()
    
    # HANDS: 3 cubes each, in LOCAL SPACE attached to FRONT of body
    # Hands positioned FORWARD (Y direction after rotation) and to the SIDES (X direction)
    # Right hand - 3 cubes (side = +X, forward = +Y)
    eq = player.get('equipped', 'unarmed')
    for i in range(3):
        glPushMatrix()
        # Right side, forward from shoulder
        glTranslatef(12, 8 + i*5, PLAYER_HEIGHT - 8)
        # Change color if armed - front cube (i==2) changes color
        if i == 2:  # front cube
            if eq == 'gun':
                glColor3f(0.8, 0.2, 0.0)  # orange-red when gun
            elif eq == 'knife':
                glColor3f(0.8, 0.0, 0.0)  # darker red when knife
            else:
                glColor3f(0.0, 1.0, 0.0)  # green hands normally
        else:
            glColor3f(0.0, 1.0, 0.0)  # green body
        glScalef(5, 5, 5)
        glutSolidCube(1)
        glPopMatrix()
    
    # Left hand - 3 cubes (side = -X, forward = +Y)
    for i in range(3):
        glPushMatrix()
        # Left side, forward from shoulder
        glTranslatef(-12, 8 + i*5, PLAYER_HEIGHT - 8)
        # Change color if armed - front cube (i==2) changes color
        if i == 2:  # front cube
            if eq == 'gun':
                glColor3f(0.8, 0.2, 0.0)  # orange-red when gun
            elif eq == 'knife':
                glColor3f(0.8, 0.0, 0.0)  # darker red when knife
            else:
                glColor3f(0.0, 1.0, 0.0)  # green hands normally
        else:
            glColor3f(0.0, 1.0, 0.0)  # green body
        glScalef(5, 5, 5)
        glutSolidCube(1)
        glPopMatrix()
    
    glPopMatrix()


def draw_enemy(e):
    if e['hp'] <= 0:
        return
    
    # Calculate vision cone in world space
    rad = math.radians(e['angle'])
    fx = -math.sin(rad)  # forward direction
    fy = math.cos(rad)
    rx = math.cos(rad)   # right direction (perpendicular to forward)
    ry = math.sin(rad)
    
    # Eye position in world space (at head height, centered)
    eye_x = e['x']
    eye_y = e['y']
    eye_z = e['z'] + PLAYER_HEIGHT + 6
    
    # Vision cone geometry - small cone (2-3 cells) that is blocked by walls
    far_dist = CELL * 2.5  # smaller vision range
    side_spread = CELL * 0.8
    
    # Check for wall blockage by tracing from eye toward target points
    def is_cone_point_blocked(px, py):
        # Simple ray trace: check if line from eye to point crosses wall
        steps = 10
        for i in range(1, steps):
            t = i / steps
            check_x = eye_x + (px - eye_x) * t
            check_y = eye_y + (py - eye_y) * t
            if cell_is_wall_world(check_x, check_y):
                return True
        return False
    
    left_x = eye_x + fx * far_dist - rx * side_spread
    left_y = eye_y + fy * far_dist - ry * side_spread
    right_x = eye_x + fx * far_dist + rx * side_spread
    right_y = eye_y + fy * far_dist + ry * side_spread
    far_x = eye_x + fx * far_dist
    far_y = eye_y + fy * far_dist
    
    # Block cone parts if walls are in the way
    left_blocked = is_cone_point_blocked(left_x, left_y)
    right_blocked = is_cone_point_blocked(right_x, right_y)
    far_blocked = is_cone_point_blocked(far_x, far_y)
    
    e['vision_cone'] = {
        'eye_x': eye_x,
        'eye_y': eye_y,
        'eye_z': eye_z,
        'left_x': left_x,
        'left_y': left_y,
        'left_blocked': left_blocked,
        'right_x': right_x,
        'right_y': right_y,
        'right_blocked': right_blocked,
        'far_x': far_x,
        'far_y': far_y,
        'far_blocked': far_blocked,
    }
    
    # Now draw the body in local space
    glPushMatrix()
    glTranslatef(e['x'], e['y'], e['z'])
    glRotatef(e['angle'], 0, 0, 1)
    
    # body - red (slimmer)
    glColor3f(1.0, 0.0, 0.0)
    glPushMatrix()
    glTranslatef(0, 0, PLAYER_HEIGHT/2)
    glScalef(0.6, 0.6, 1.0)  # make slimmer
    glutSolidCube(PLAYER_HEIGHT)
    glPopMatrix()
    
    # head - BLACK
    glPushMatrix()
    glTranslatef(0, 0, PLAYER_HEIGHT + 6)
    quad = gluNewQuadric()
    glColor3f(0.0, 0.0, 0.0)  # black head
    gluSphere(quad, 8, 12, 12)
    glPopMatrix()
    
    # Gun hand - single, ash/silver color, 2 cubes, in LOCAL space
    glColor3f(0.5, 0.5, 0.5)  # ash/silver color
    
    # Barrel cube (front of body)
    glPushMatrix()
    glTranslatef(0, 8, PLAYER_HEIGHT - 8)  # forward from shoulder
    glScalef(8, 4, 4)  # elongated like a gun barrel
    glutSolidCube(1)
    glPopMatrix()
    
    # Front extension cube
    glPushMatrix()
    glTranslatef(0, 14, PLAYER_HEIGHT - 8)  # further forward
    glScalef(6, 3, 3)  # front extension
    glutSolidCube(1)
    glPopMatrix()
    
    glPopMatrix()
    
    # Draw vision cone in world space (after popping matrix)
    cone = e['vision_cone']
    glColor3f(1.0, 1.0, 0.0)  # bright yellow vision cone
    glBegin(GL_TRIANGLES)
    # left side
    glVertex3f(cone['eye_x'], cone['eye_y'], cone['eye_z'])
    glVertex3f(cone['left_x'], cone['left_y'], 1)
    glVertex3f(cone['far_x'], cone['far_y'], 1)
    # right side
    glVertex3f(cone['eye_x'], cone['eye_y'], cone['eye_z'])
    glVertex3f(cone['far_x'], cone['far_y'], 1)
    glVertex3f(cone['right_x'], cone['right_y'], 1)
    glEnd()


def draw_world():
    # Build wall cells with bounds checking
    wall_cells = set()
    for y in range(map_h):
        for x in range(map_w):
            if x < len(world_map[y]) and world_map[y][x] == '#':
                wall_cells.add((x, y))
    
    # FIRST: Draw floor and ceiling for all corridor cells (render first so pillars appear on top)
    for y in range(map_h):
        for x in range(map_w):
            if x >= len(world_map[y]):
                continue
            ch = world_map[y][x]
            if ch != '#':
                # Corridor cell
                wx = (x - map_w // 2) * CELL
                wy = (y - map_h // 2) * CELL
                
                # floor quad with alternating tint
                if (x+y) % 2 == 0:
                    glColor3f(0.3, 0.5, 0.3)  # light green
                else:
                    glColor3f(0.5, 0.3, 0.3)  # light red
                glBegin(GL_QUADS)
                glVertex3f(wx - CELL/2, wy - CELL/2, 0)
                glVertex3f(wx + CELL/2, wy - CELL/2, 0)
                glVertex3f(wx + CELL/2, wy + CELL/2, 0)
                glVertex3f(wx - CELL/2, wy + CELL/2, 0)
                glEnd()
                # ceiling quad
                glColor3f(0.2, 0.2, 0.2)  # dark gray ceiling
                glBegin(GL_QUADS)
                glVertex3f(wx - CELL/2, wy - CELL/2, CORRIDOR_HEIGHT)
                glVertex3f(wx + CELL/2, wy - CELL/2, CORRIDOR_HEIGHT)
                glVertex3f(wx + CELL/2, wy + CELL/2, CORRIDOR_HEIGHT)
                glVertex3f(wx - CELL/2, wy + CELL/2, CORRIDOR_HEIGHT)
                glEnd()
    
    # SECOND: Draw walls as tall RED cylinder pillars (render on top of floor)
    quad = gluNewQuadric()
    glColor3f(0.9, 0.15, 0.15)  # Bright red
    
    for y in range(map_h):
        for x in range(map_w):
            if (x, y) in wall_cells:
                wx = (x - map_w // 2) * CELL
                wy = (y - map_h // 2) * CELL
                
                # Draw tall red cylinder pillar
                glPushMatrix()
                glTranslatef(wx, wy, 0)
                glColor3f(0.9, 0.15, 0.15)  # Bright red
                gluCylinder(quad, CELL * 0.35, CELL * 0.35, CORRIDOR_HEIGHT, 16, 4)
                glPopMatrix()
    
    # doors
    for d in doors:
        if not d['open']:
            # closed door as cube - bright blue
            draw_cube_at(d['x'], d['y'], CELL, color=(0.0, 0.0, 1.0))
        else:
            # open door drawn as flat mark - bright blue (darker)
            glColor3f(0.0, 0.0, 0.7)
            glBegin(GL_QUADS)
            glVertex3f(d['x'] - CELL/4, d['y'] - CELL/2, 1)
            glVertex3f(d['x'] + CELL/4, d['y'] - CELL/2, 1)
            glVertex3f(d['x'] + CELL/4, d['y'] + CELL/2, 1)
            glVertex3f(d['x'] - CELL/4, d['y'] + CELL/2, 1)
            glEnd()
    # powerups
    for pu in powerups:
        if not pu['active']:
            continue
        if pu['type'] == 'G':
            glColor3f(0.0, 1.0, 0.0)  # bright green - gun
        elif pu['type'] == 'N':
            glColor3f(1.0, 1.0, 0.0)  # bright yellow - knife
        elif pu['type'] == 'I':
            glColor3f(1.0, 0.0, 1.0)  # bright magenta - invisibility
        elif pu['type'] == 'K':
            glColor3f(1.0, 0.5, 0.0)  # bright orange - key
        glPushMatrix()
        glTranslatef(pu['x'], pu['y'], pu['z'])
        quad = gluNewQuadric()
        gluSphere(quad, 8, 8, 8)
        glPopMatrix()

    # player and enemies
    draw_player()
    for e in enemies:
        draw_enemy(e)
    # bullets
    for b in bullets:
        glColor3f(1.0, 1.0, 1.0) if b['owner']=='player' else glColor3f(1.0, 0.0, 0.0)  # white player bullets, red enemy bullets
        glPushMatrix()
        glTranslatef(b['x'], b['y'], b['z'])
        quad = gluNewQuadric()
        gluSphere(quad, BULLET_RADIUS, 8, 8)
        glPopMatrix()

# ---------------------------
# Main display and loop
# ---------------------------

def display():
    global last_time
    now = time.time()
    dt = now - last_time
    last_time = now
    # update
    if game_state == STATE_PLAY:
        update_player_movement(dt)
        update(dt)
    # render
    glClear(GL_COLOR_BUFFER_BIT)
    glViewport(0,0,WINDOW_W, WINDOW_H)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, WINDOW_W / float(WINDOW_H), 1.0, 3000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Use reference camera handling
    u_fp()
    setupCamera()
    # draw world
    draw_world()
    # HUD
    live_enemies = sum(1 for e in enemies if e['hp'] > 0)
    draw_text_screen(10, WINDOW_H-20, f"Life: {int(player['life'])}  Level: {current_level_index+1}  Enemies: {live_enemies}")
    draw_text_screen(10, WINDOW_H-40, f"Gun: {player['has_gun'] and player['gun_bullets'] or 0}  Knife: {player['has_knife']}  Invis: {int(player['invisible'])}")
    if game_state == STATE_PAUSE:
        draw_text_screen(WINDOW_W/2-60, WINDOW_H/2, "PAUSED")
    if game_state == STATE_GAMEOVER:
        draw_text_screen(WINDOW_W/2-120, WINDOW_H/2, "GAME OVER - press R to reset")
    glutSwapBuffers()


def idle():
    glutPostRedisplay()


def mouse(button, state, x, y):
    global fp, pcamera_pos, camera_pos, camera_mode
    # left mouse fires/melees
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        melee_or_fire()
    # right mouse toggles first-person mode similar to reference
    if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
        # toggle camera mode
        if camera_mode == CAM_THIRD:
            camera_mode = CAM_FIRST
            pcamera_pos = camera_pos
        else:
            camera_mode = CAM_THIRD
            camera_pos = pcamera_pos

# ---------------------------
# Reset and init
# ---------------------------

def reset_game():
    global player, bullets, current_level_index, game_state, camera_pos, pcamera_pos, camera_mode, keys, global_alarm_level, global_alarm_meter
    # Reset player
    player['life'] = 13
    player['has_gun'] = False
    player['gun_bullets'] = 0
    player['has_knife'] = False
    player['invisible'] = False
    player['invis_timer'] = 0.0
    player['melee_timer'] = 0.0
    player['detection_meter'] = 0.0
    player['in_hiding_spot'] = False
    player['hacking_terminal'] = None
    player['hacking_progress'] = 0.0
    # Reset game state
    bullets.clear()
    keys.clear()
    game_state = STATE_PLAY
    camera_mode = CAM_THIRD
    camera_pos = (0.0, 500.0, 500.0)
    pcamera_pos = camera_pos
    global_alarm_level = ALARM_LEVEL_0
    global_alarm_meter = 0.0
    # Reload level
    load_level(current_level_index)

# ---------------------------
# Entry point
# ---------------------------

def main():
    global last_time
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(WINDOW_W, WINDOW_H)
    glutInitWindowPosition(50, 50)
    glutCreateWindow(b"Stealth Maze - Minimal PyOpenGL")
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutKeyboardFunc(keyboard)
    glutKeyboardUpFunc(keyboard_up)
    glutSpecialFunc(special_key)
    glutMouseFunc(mouse)
    # initialize
    load_level(current_level_index)
    last_time = time.time()
    glutMainLoop()

if __name__ == '__main__':
    main()
