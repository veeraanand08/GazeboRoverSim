import heapq
import math
import random
import subprocess
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# sdf string for the little red dot we spawn at each waypoint so you can
# actually see where the robot is supposed to go in the gazebo gui
MARKER_SDF = """<sdf version="1.9">
<model name="{name}">
  <static>true</static>
  <link name="link">
    <visual name="dot">
      <pose>0 0 0.01 0 0 0</pose>
      <geometry><cylinder><radius>0.4</radius><length>0.02</length></cylinder></geometry>
      <material>
        <ambient>1 0 0 1</ambient>
        <diffuse>1 0 0 1</diffuse>
        <emissive>0.3 0 0 1</emissive>
      </material>
    </visual>
  </link>
</model>
</sdf>"""


# hardcoded rock positions from the world file (x, y, radius). we use this
# list ONLY when picking where to put the random waypoints so they dont
# spawn on top of a rock. the robot itself does NOT get to use this list,
# it has to find the rocks with its own lidar like a real robot would
ROCK_LIST = [
    (3.74, 0.66, 0.6),
    (-2.76, 3.29, 1.2),
    (-1.37, -3.76, 0.7),
]


def get_yaw(q):
    # converts the quaternion from odometry into a simple yaw angle (radians)
    # this is just the standard formula for yaw from a quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def fix_angle(a):
    # keeps an angle between -pi and pi so the math doesnt break when
    # comparing headings that wrap around (like 179 deg vs -179 deg)
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class State(Enum):
    MAPPING = auto()
    GOING_TO_WP = auto()
    STUCK = auto()
    DONE = auto()


class RoverNavNode(Node):
    """
    this node does the whole navigation thing for the project:

    1. first it drives around a little bit (a square pattern) so the lidar
       can see the rocks from a few different angles and we can build a map
       (occupancy grid). this is the "mapping" step
    2. once that initial loop is done we use A* to find a path to the first
       waypoint, simplify it so its not just a bunch of jagged grid steps,
       and start driving it point by point
    3. UNLIKE before, the map doesnt freeze after that -- the lidar keeps
       adding new hits to the grid the whole time were driving, and every
       couple seconds we quietly re-run A* to the current waypoint using
       whatever the grid looks like NOW. if a new/moved obstacle shows up
       somewhere on our path, the next replan just routes around it. (this
       is NOT real SLAM btw -- real SLAM has to figure out where the robot
       IS using the scans, because theres no ground truth position sensor.
       we just get perfect x/y/yaw for free from gazebo's odometry, so all
       were really doing is continuous mapping + replanning, not full
       localization + mapping. figured that was worth being honest about)
    4. when it gets to a waypoint we plan a brand new path to the next one
       from wherever we currently are
    5. if the robot isnt making any progress for a while (stuck on
       something) it backs up, turns a random way, and tries again. if it
       fails enough times it gives up on that waypoint and moves to the
       next one so the whole run doesnt get stuck forever
    """

    def __init__(self):
        super().__init__('astar_nav_node')

        # ---- all our tunable numbers, as ROS params so we can change them
        # without recompiling ----
        self.declare_parameter('num_waypoints', 3)
        self.declare_parameter('area_min', -9.0)
        self.declare_parameter('area_max', 9.0)
        self.declare_parameter('spawn_exclusion_radius', 3.0)
        self.declare_parameter('min_waypoint_separation', 3.0)
        self.declare_parameter('grid_resolution', 0.25)
        self.declare_parameter('robot_radius', 0.95)
        self.declare_parameter('goal_tolerance', 0.4)
        self.declare_parameter('max_linear_speed', 2.4)
        self.declare_parameter('max_angular_speed', 3.6)
        self.declare_parameter('heading_kp', 1.5)
        self.declare_parameter('heading_ki', 0.05)
        self.declare_parameter('heading_integral_limit', 1.0)
        self.declare_parameter('stuck_timeout', 6.0)
        self.declare_parameter('stuck_move_eps', 0.15)
        self.declare_parameter('recovery_duration', 2.0)
        self.declare_parameter('max_recovery_attempts', 3)
        self.declare_parameter('replan_interval', 2.0)
        self.declare_parameter('seed', -1)
        self.declare_parameter('world_name', 'obstacle_world')

        num_wp = self.get_parameter('num_waypoints').value
        self.xmin = self.get_parameter('area_min').value
        self.xmax = self.get_parameter('area_max').value
        self.spawn_radius = self.get_parameter('spawn_exclusion_radius').value
        self.wp_gap = self.get_parameter('min_waypoint_separation').value
        self.res = self.get_parameter('grid_resolution').value
        self.robot_r = self.get_parameter('robot_radius').value
        self.tol = self.get_parameter('goal_tolerance').value
        self.max_v = self.get_parameter('max_linear_speed').value
        self.max_w = self.get_parameter('max_angular_speed').value
        self.kp = self.get_parameter('heading_kp').value
        self.ki = self.get_parameter('heading_ki').value
        self.i_limit = self.get_parameter('heading_integral_limit').value
        self.stuck_time_limit = self.get_parameter('stuck_timeout').value
        self.stuck_dist_eps = self.get_parameter('stuck_move_eps').value
        self.recover_time = self.get_parameter('recovery_duration').value
        self.max_recover_tries = self.get_parameter('max_recovery_attempts').value
        self.replan_interval = self.get_parameter('replan_interval').value
        seed_val = self.get_parameter('seed').value
        self.world_name = self.get_parameter('world_name').value

        if seed_val >= 0:
            random.seed(seed_val)

        # build the empty grid. grid_n is how many cells wide/tall the
        # square map is. we store it as one big flat list instead of a 2d
        # list because its a bit simpler to index (row * width + col)
        self.grid_n = int(round((self.xmax - self.xmin) / self.res)) + 1
        self.grid = [False] * (self.grid_n * self.grid_n)

        self.wp_list = self.make_random_waypoints(num_wp)
        if len(self.wp_list) < num_wp:
            self.get_logger().warn(
                f'only found {len(self.wp_list)} good spots out of {num_wp} requested waypoints'
            )
        self.wp_num = 0  # index of the waypoint we are currently trying to reach
        self.get_logger().info(f'waypoints: {self.wp_list}')

        self.state = State.MAPPING

        # this is just a hardcoded square path the robot drives at the very
        # start so the lidar gets to see stuff from more than 1 spot. its
        # not smart/reactive at all, just forward-turn-forward-turn etc
        self.explore_speed = 0.8
        self.explore_turn_speed = 1.6
        self.explore_steps = [
            ('fwd', 1.5),
            ('turn', math.pi / 2),
            ('fwd', 1.5),
            ('turn', math.pi / 2),
            ('fwd', 1.5),
            ('turn', math.pi / 2),
            ('fwd', 1.5),
            ('turn', math.pi / 2),
        ]
        self.explore_step_num = 0
        self.step_start_pos = None
        self.step_start_yaw = None

        # stuff for checking if we are stuck
        self.last_move_time = time.time()
        self.last_pos = (0.0, 0.0)
        self.unstuck_until = 0.0
        self.unstuck_turn_dir = 1.0
        self.stuck_count = 0

        self.got_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.got_scan = False
        self.ranges = []
        self.angle_min = 0.0
        self.angle_inc = 0.0
        self.max_range = 10.0

        self.path = None  # list of (x,y) points to the CURRENT waypoint
        self.path_i = 0
        self.last_replan_time = time.time()

        self.i_term = 0.0
        self.dt = 0.1  # how often our timer runs, in seconds

        self.vel_pub = self.create_publisher(Twist, '/model/waypoint_bot/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/waypoint_status', 10)
        # transient_local basically means "keep the last message around" so
        # if you open rviz after the map already got published you still
        # get it
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/astar_map', map_qos)
        self.create_subscription(Odometry, '/model/waypoint_bot/odometry', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/lidar', self.scan_callback, 10)

        self.make_wp_markers()

        self.timer = self.create_timer(self.dt, self.main_loop)

    # ------------------------------------------------------------------
    # waypoint generation stuff
    # ------------------------------------------------------------------

    def make_random_waypoints(self, how_many):
        # picks random x,y spots on the map and throws them out if they
        # land somewhere bad (on a rock, too close to spawn, or too close
        # to another waypoint we already picked). keeps trying until it
        # gets enough good ones or gives up after 1000 tries total
        edge_margin = 0.75
        buffer = self.robot_r + 0.3

        # list of "no go" circles: (center x, center y, radius). starts
        # with the rocks + the spawn area, then we add more as we pick
        # waypoints so they dont clump up next to each other
        no_go = [(rx, ry, rr + buffer) for rx, ry, rr in ROCK_LIST]
        no_go.append((0.0, 0.0, self.spawn_radius))

        picked = []
        tries = 0
        while len(picked) < how_many and tries < 1000:
            tries += 1
            x = random.uniform(self.xmin + edge_margin, self.xmax - edge_margin)
            y = random.uniform(self.xmin + edge_margin, self.xmax - edge_margin)

            bad_spot = False
            for cx, cy, r in no_go:
                if math.hypot(x - cx, y - cy) <= r:
                    bad_spot = True
                    break
            if bad_spot:
                continue

            picked.append((x, y))
            no_go.append((x, y, self.wp_gap))
        return picked

    def send_status(self, txt):
        self.status_pub.publish(String(data=txt))

    def send_map(self):
        # publishes our occupancy grid as a normal ros OccupancyGrid msg so
        # you can look at it in rviz2 (add by topic -> /astar_map -> Map,
        # and set fixed frame to odom)
        m = OccupancyGrid()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.info.resolution = self.res
        m.info.width = self.grid_n
        m.info.height = self.grid_n
        m.info.origin.position.x = self.xmin
        m.info.origin.position.y = self.xmin
        m.info.origin.orientation.w = 1.0
        # 100 = blocked cell, 0 = free cell (this isnt using -1/unknown at
        # all, we just treat everything we havent seen as free)
        m.data = [100 if c else 0 for c in self.grid]
        self.map_pub.publish(m)
        self.get_logger().info('published map to /astar_map')

    def make_wp_markers(self):
        # spawns the little red dots in gazebo so we can see the waypoints.
        # this has nothing to do with navigation, its purely visual.
        # gazebo takes a sec to boot up so we just keep retrying the spawn
        # command a bunch of times instead of only trying once
        max_tries = 20
        wait_between = 1.5
        for idx, (wx, wy) in enumerate(self.wp_list):
            marker_name = f'wp_marker_{idx}'
            sdf_str = MARKER_SDF.format(name=marker_name)
            worked = False
            err_msg = 'unknown error'
            for t in range(max_tries):
                try:
                    r = subprocess.run(
                        [
                            'ros2', 'run', 'ros_gz_sim', 'create',
                            '-world', self.world_name,
                            '-name', marker_name,
                            '-x', str(wx), '-y', str(wy), '-z', '0',
                            '-string', sdf_str,
                        ],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode == 0:
                        worked = True
                        break
                    err_msg = r.stderr.strip()
                except subprocess.TimeoutExpired:
                    # if this happens dont let it crash the whole node, just
                    # count it as a failed try and go again
                    err_msg = 'timed out, gazebo probably still starting up'
                if t < max_tries - 1:
                    time.sleep(wait_between)
            if worked:
                self.get_logger().info(f'spawned marker {idx} at ({wx:.2f}, {wy:.2f})')
            else:
                self.get_logger().warn(f'couldnt spawn marker {idx}, gave up after {max_tries} tries: {err_msg}')

    def odom_callback(self, msg):
        self.got_odom = True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = get_yaw(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        self.got_scan = True
        self.ranges = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_inc = msg.angle_increment
        self.max_range = msg.range_max

    # ------------------------------------------------------------------
    # grid / occupancy map helpers
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        # turns a real world x,y coordinate into a grid cell (row, col)
        gx = int(round((x - self.xmin) / self.res))
        gy = int(round((y - self.xmin) / self.res))
        return self.clamp_cell(gx, gy)

    def grid_to_world(self, gx, gy):
        # opposite of world_to_grid, turns a cell back into an x,y point
        # (this will be the middle-ish of the cell basically)
        return (self.xmin + gx * self.res, self.xmin + gy * self.res)

    def clamp_cell(self, gx, gy):
        # makes sure we dont go outside the bounds of the grid array
        n = self.grid_n
        gx = max(0, min(n - 1, gx))
        gy = max(0, min(n - 1, gy))
        return (gx, gy)

    def is_blocked(self, gx, gy):
        return self.grid[gy * self.grid_n + gx]

    def block_area_around(self, x, y):
        # marks a circle of cells as blocked around this point (not just
        # the single cell) so the robot keeps some distance away from
        # obstacles instead of just avoiding the exact hit point
        cx, cy = self.world_to_grid(x, y)
        n = self.grid_n
        r_cells = max(1, int(round(self.robot_r / self.res)))
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if math.hypot(dx, dy) * self.res > self.robot_r:
                    continue  # outside the circle, skip it
                gx = cx + dx
                gy = cy + dy
                if 0 <= gx < n and 0 <= gy < n:
                    self.grid[gy * n + gx] = True

    def scan_to_grid(self):
        # goes through the latest lidar scan and marks every hit as
        # blocked in our grid. gets called every tick no matter what state
        # we're in, so the map keeps growing/updating the whole run, not
        # just during the initial mapping loop
        if self.angle_inc == 0.0:
            return
        for idx, dist in enumerate(self.ranges):
            if math.isinf(dist) or math.isnan(dist) or dist <= 0.0 or dist >= self.max_range:
                continue  # bad reading, ignore it
            beam_angle = self.yaw + self.angle_min + idx * self.angle_inc
            hit_x = self.x + dist * math.cos(beam_angle)
            hit_y = self.y + dist * math.sin(beam_angle)
            self.block_area_around(hit_x, hit_y)

    # ------------------------------------------------------------------
    # the actual A* algorithm
    # ------------------------------------------------------------------

    def run_astar(self, start, goal):
        # classic grid based A*, 8 directions (so it can move diagonally
        # too, not just up/down/left/right)
        n = self.grid_n
        if self.is_blocked(*start) or self.is_blocked(*goal):
            return None  # cant even start or end on a blocked cell

        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        def h(a, b):
            # heuristic - just straight line distance to the goal
            return math.hypot(a[0] - b[0], a[1] - b[1])

        pq = [(h(start, goal), start)]  # priority queue, sorted by lowest cost first
        came_from = {}
        cost_so_far = {start: 0.0}
        done = set()

        while pq:
            _, current = heapq.heappop(pq)
            if current in done:
                continue
            done.add(current)

            if current == goal:
                # we found it! walk backwards through came_from to build
                # the actual path, then flip it around so it goes start->goal
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for mx, my in moves:
                nx = current[0] + mx
                ny = current[1] + my
                if nx < 0 or nx >= n or ny < 0 or ny >= n:
                    continue
                if self.is_blocked(nx, ny):
                    continue
                neighbor = (nx, ny)
                new_cost = cost_so_far[current] + math.hypot(mx, my)
                if new_cost < cost_so_far.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost + h(neighbor, goal), neighbor))

        return None  # ran out of stuff to check, no path exists

    def can_see_between(self, cell_a, cell_b):
        # checks if theres a clear straight line between 2 grid cells (no
        # obstacles in between). used for cutting corners out of the raw
        # A* path so it doesnt look all jagged/staircase-y
        x0, y0 = self.grid_to_world(*cell_a)
        x1, y1 = self.grid_to_world(*cell_b)
        d = math.hypot(x1 - x0, y1 - y0)
        num_steps = max(1, int(d / (self.res * 0.5)))
        for i in range(num_steps + 1):
            t = i / num_steps
            px = x0 + (x1 - x0) * t
            py = y0 + (y1 - y0) * t
            gx, gy = self.world_to_grid(px, py)
            if self.is_blocked(gx, gy):
                return False
        return True

    def shorten_path(self, cell_path):
        # takes the raw zig-zag path from A* (which just moves cell by
        # cell) and cuts out anything we dont need by connecting far apart
        # points directly if theres nothing blocking a straight line
        # between them. makes the path way shorter/smoother
        if not cell_path:
            return []
        result = [cell_path[0]]
        i = 0
        while i < len(cell_path) - 1:
            j = len(cell_path) - 1
            while j > i + 1 and not self.can_see_between(cell_path[i], cell_path[j]):
                j -= 1
            result.append(cell_path[j])
            i = j
        return [self.grid_to_world(gx, gy) for gx, gy in result]

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------

    def stop_robot(self):
        self.vel_pub.publish(Twist())

    def main_loop(self):
        # runs every self.dt seconds, just checks what state we're in and
        # calls the right function
        if not self.got_odom or not self.got_scan:
            return  # dont do anything until we have real sensor data

        # keep the map updating live, every tick, no matter what state
        # we're in -- this is what makes it "adaptive" instead of a one
        # time snapshot
        self.scan_to_grid()

        if self.state == State.MAPPING:
            self.do_mapping()
        elif self.state == State.GOING_TO_WP:
            self.do_following()
        elif self.state == State.STUCK:
            self.do_recovery()
        else:
            self.stop_robot()

    def do_recovery(self):
        # back up a bit and spin, hopefully that gets us unstuck. after
        # self.recover_time seconds go back to trying to drive normally
        now = time.time()
        if now >= self.unstuck_until:
            self.state = State.GOING_TO_WP
            self.last_move_time = now
            self.last_pos = (self.x, self.y)
            self.stop_robot()
            return
        cmd = Twist()
        cmd.linear.x = -0.5
        cmd.angular.z = self.unstuck_turn_dir * self.max_w * 0.4
        self.vel_pub.publish(cmd)

    def do_mapping(self):
        if self.explore_step_num >= len(self.explore_steps):
            # done with our little square drive, now we can actually plan
            self.stop_robot()
            self.get_logger().info('done exploring, map is ready')
            self.send_map()
            self.plan_path_to_wp()
            self.state = State.GOING_TO_WP
            return

        if self.step_start_pos is None:
            self.step_start_pos = (self.x, self.y)
            self.step_start_yaw = self.yaw
            step_type, amt = self.explore_steps[self.explore_step_num]
            self.get_logger().info(f'exploring step {self.explore_step_num + 1}/{len(self.explore_steps)}: {step_type} {amt:.2f}')
            self.send_status(f'exploring ({self.explore_step_num + 1}/{len(self.explore_steps)})')

        step_type, amt = self.explore_steps[self.explore_step_num]
        cmd = Twist()

        if step_type == 'fwd':
            dist_gone = math.hypot(self.x - self.step_start_pos[0], self.y - self.step_start_pos[1])
            if dist_gone >= amt:
                self.explore_step_num += 1
                self.step_start_pos = None
                self.stop_robot()
                return
            cmd.linear.x = self.explore_speed
            cmd.angular.z = 0.0
        else:
            # turning step
            turned_amt = abs(fix_angle(self.yaw - self.step_start_yaw))
            if turned_amt >= amt - 0.05:
                self.explore_step_num += 1
                self.step_start_pos = None
                self.stop_robot()
                return
            cmd.linear.x = 0.0
            cmd.angular.z = self.explore_turn_speed

        self.vel_pub.publish(cmd)

    def plan_path_to_wp(self):
        # tries to A* to whatever waypoint we're currently on
        # (self.wp_num). if that one is impossible to reach, just skip it
        # and try the next one, and keep doing that till one works or we
        # run out of waypoints
        total = len(self.wp_list)
        while self.wp_num < total:
            goal_xy = self.wp_list[self.wp_num]
            start_cell = self.world_to_grid(self.x, self.y)
            goal_cell = self.world_to_grid(*goal_xy)
            raw_path = self.run_astar(start_cell, goal_cell)

            if raw_path is None:
                self.get_logger().warn(f'no path to waypoint {self.wp_num}, skipping it')
                self.send_status(f'waypoint {self.wp_num + 1}/{total} unreachable - skipped')
                self.wp_num += 1
                continue  # try the next waypoint instead

            nice_path = self.shorten_path(raw_path)
            nice_path[-1] = goal_xy  # force the last point to be exactly the waypoint (avoids rounding)
            self.path = nice_path
            self.path_i = 0
            self.i_term = 0.0
            self.stuck_count = 0
            self.last_move_time = time.time()
            self.last_pos = (self.x, self.y)
            self.last_replan_time = time.time()
            self.get_logger().info(f'found path to wp {self.wp_num}, {len(nice_path)} points')
            self.send_status(f'waypoint {self.wp_num + 1}/{total} | path found, following')
            return

        # if we get here, we ran through every remaining waypoint and none
        # of them worked
        self.path = None

    def try_replan(self):
        # this is the "adaptive" part. re-runs A* from wherever we are
        # right now to the SAME waypoint were already going for, but using
        # the grid as it looks at THIS moment (which mightve picked up new
        # obstacle hits since we last planned). if it finds a path we swap
        # it in. if it doesnt find one, we just keep driving the old path
        # and let the normal stuck-detection deal with it if it really is
        # blocked -- dont want 1 bad/noisy scan to just stop the robot
        goal_xy = self.wp_list[self.wp_num]
        start_cell = self.world_to_grid(self.x, self.y)
        goal_cell = self.world_to_grid(*goal_xy)
        raw_path = self.run_astar(start_cell, goal_cell)

        if raw_path is None:
            self.get_logger().warn(f'replan failed for waypoint {self.wp_num} (map changed?), sticking with old path')
            return

        nice_path = self.shorten_path(raw_path)
        nice_path[-1] = goal_xy
        self.path = nice_path
        self.path_i = 0
        self.send_map()  # let rviz see the updated map too

    def do_following(self):
        total = len(self.wp_list)

        if self.wp_num >= total:
            # we've done all the waypoints
            if self.state != State.DONE:
                self.get_logger().info('all done! finished every waypoint')
                self.send_status(f'mission complete: {total}/{total} waypoints visited')
                self.state = State.DONE
            self.stop_robot()
            return

        if self.path is None:
            self.stop_robot()
            return

        # every replan_interval seconds, try re-planning to the same
        # waypoint using the live-updated map -- this is what lets it
        # react to something it didnt see during the original mapping loop
        now_for_replan = time.time()
        if now_for_replan - self.last_replan_time > self.replan_interval:
            self.last_replan_time = now_for_replan
            self.try_replan()

        goal_x, goal_y = self.path[self.path_i]
        dist = math.hypot(goal_x - self.x, goal_y - self.y)

        if dist < self.tol:
            if self.path_i < len(self.path) - 1:
                # this was just a point along the way, not the actual
                # waypoint yet, so move on to the next point in the path
                self.path_i += 1
                return

            # if we're here, path_i was the LAST point in the path, meaning
            # this is actually the waypoint itself
            num = self.wp_num + 1
            self.get_logger().info(f'reached waypoint {self.wp_num} at ({goal_x:.2f}, {goal_y:.2f})')
            self.send_status(f'waypoint {num}/{total} reached!')
            self.wp_num += 1
            if self.wp_num >= total:
                self.get_logger().info('all done! finished every waypoint')
                self.send_status(f'mission complete: {total}/{total} waypoints visited')
                self.state = State.DONE
                self.stop_robot()
            else:
                self.plan_path_to_wp()  # go plan the next one
            return

        # ---- check if we're stuck ----
        now = time.time()
        moved_dist = math.hypot(self.x - self.last_pos[0], self.y - self.last_pos[1])
        if moved_dist > self.stuck_dist_eps:
            # we moved a decent amount recently so we're probably fine
            self.last_move_time = now
            self.last_pos = (self.x, self.y)
        elif now - self.last_move_time > self.stuck_time_limit:
            # havent moved in a while, try to recover
            self.stuck_count += 1
            if self.stuck_count > self.max_recover_tries:
                # tried too many times, just give up on this waypoint
                num = self.wp_num + 1
                self.get_logger().warn(f'waypoint {self.wp_num} seems impossible, tried recovering {self.stuck_count - 1} times. skipping')
                self.send_status(f'waypoint {num}/{total} skipped (stuck)')
                self.wp_num += 1
                if self.wp_num >= total:
                    self.state = State.DONE
                    self.stop_robot()
                else:
                    self.plan_path_to_wp()
                return
            self.get_logger().warn(f'stuck near waypoint {self.wp_num}, trying to recover ({self.stuck_count}/{self.max_recover_tries})')
            self.send_status(f'waypoint {self.wp_num + 1}/{total} | recovering ({self.stuck_count}/{self.max_recover_tries})')
            self.unstuck_turn_dir = random.choice([-1.0, 1.0])
            self.unstuck_until = now + self.recover_time
            self.state = State.STUCK
            return

        # ---- normal driving, simple P controller on heading ----
        # figure out what angle we need to be facing to reach the target
        want_angle = math.atan2(goal_y - self.y, goal_x - self.x)
        err = fix_angle(want_angle - self.yaw)

        # small integral term too so it doesnt have a steady state error,
        # clamped so it cant build up forever
        self.i_term = max(-self.i_limit, min(self.i_limit, self.i_term + err * self.dt))
        turn_speed = self.kp * err + self.ki * self.i_term
        turn_speed = max(-self.max_w, min(self.max_w, turn_speed))

        # if the angle we need to turn is big, just stop and turn in place
        # first instead of also driving forward. otherwise it "cuts the
        # corner" while turning and can clip an obstacle even though A*
        # planned a path that avoids it
        turn_in_place_cutoff = 0.35  # about 20 degrees
        if abs(err) > turn_in_place_cutoff:
            fwd_speed = 0.0
        else:
            fwd_speed = self.max_v * (1.0 - abs(err) / turn_in_place_cutoff)

        cmd = Twist()
        cmd.linear.x = fwd_speed
        cmd.angular.z = turn_speed
        self.vel_pub.publish(cmd)

        self.send_status(
            f'waypoint {self.wp_num + 1}/{total} | going to point {self.path_i + 1}/{len(self.path)} | {dist:.1f}m away'
        )


def main(args=None):
    rclpy.init(args=args)
    node = RoverNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
