from SMP.motion_planner.node import Node, CostNode
from SMP.motion_planner.queue import *
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization
from abc import ABC
from typing import Tuple, Union, Dict
import sys
import copy
import numpy as np

sys.path.append('../')


def search_dict(dictionary, node_current):
    for key, value in dictionary.items():
        if node_current.get_position() == value[0][0].get_position():
            return key
    return -1


class SequentialSearch(SearchBaseClass, ABC):
    """
    Abstract class for search motion planners.
    """

    # declaration of class variables
    path_fig: Union[str, None]

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

    def initialize_search(self, time_pause, cost=True):
        """
        initializes the visualizer
        returns the initial node
        """
        self.list_status_nodes = []
        self.dict_node_status: Dict[int, Tuple] = {}
        self.time_pause = time_pause
        self.visited_nodes = []

        # first node
        if cost:
            node_initial = CostNode(list_paths=[[self.state_initial]],
                                    list_primitives=[self.motion_primitive_initial], depth_tree=0, cost=0)
        else:
            node_initial = Node(list_paths=[[self.state_initial]],
                                list_primitives=[self.motion_primitive_initial], depth_tree=0)

        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem,
                              self.config_plot, self.path_fig)

        self.dict_node_status = update_visualization(primitive=node_initial.list_paths[-1],
                                                     status=MotionPrimitiveStatus.IN_FRONTIER,
                                                     dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(self.list_status_nodes), time_pause=self.time_pause)
        self.list_status_nodes.append(copy.copy(self.dict_node_status))
        return node_initial

    def take_step(self, successor, node_current, cost=True):
        """
        Visualizes the step of a successor and checks if it collides with either an obstacle or a boundary
        cost is equal to the cost function up until this node
        Returns collision boolean and the child node if it does not collide
        """
        # translate and rotate motion primitive to current position
        list_primitives_current = copy.copy(node_current.list_primitives)
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        list_primitives_current.append(successor)
        self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
        if cost:
            child = CostNode(list_paths=self.path_new,
                                 list_primitives=list_primitives_current,
                                 depth_tree=node_current.depth_tree + 1,
                                 cost=self.cost_function(node_current))
        else:
            child = Node(list_paths=self.path_new, list_primitives=list_primitives_current,
                         depth_tree=node_current.depth_tree + 1)

        # check for collision, skip if is not collision-free
        if not self.is_collision_free(path_translated):
            position = self.path_new[-1][-1].position.tolist()
            self.list_status_nodes, self.dict_node_status, self.visited_nodes = self.plot_colliding_primitives(current_node=node_current,
                                                                                           path_translated=path_translated,
                                                                                           node_status=self.dict_node_status,
                                                                                           list_states_nodes=self.list_status_nodes,
                                                                                           time_pause=self.time_pause,
                                                                                           visited_nodes=self.visited_nodes)
            self.visited_nodes.append(position)
            return True, child
        return False, child

    def update_visuals(self, frontier_flag, node_current):
        """
        Visualizes a step on plot
        """
        if frontier_flag == "currently explored":
            self.dict_node_status = update_visualization(primitive=node_current.list_paths[-1],
                                                         status=MotionPrimitiveStatus.CURRENTLY_EXPLORED,
                                                         dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(self.list_status_nodes), time_pause=self.time_pause)
            self.list_status_nodes.append(copy.copy(self.dict_node_status))
        elif frontier_flag == "explored":
            self.dict_node_status = update_visualization(primitive=node_current.list_paths[-1],
                                                         status=MotionPrimitiveStatus.EXPLORED,
                                                         dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(self.list_status_nodes), time_pause=self.time_pause)
            self.list_status_nodes.append(copy.copy(self.dict_node_status))
        else:
            self.dict_node_status = update_visualization(primitive=node_current.list_paths[-1],
                                                         status=MotionPrimitiveStatus.IN_FRONTIER,
                                                         dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(self.list_status_nodes), time_pause=self.time_pause)
            self.list_status_nodes.append(copy.copy(self.dict_node_status))


    def goal_reached(self, successor, node_current):
        """
        Checks if the goal is reached.
        Returns True/False if goal is reached
        """
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        # goal test
        if self.reached_goal(path_translated):
            # goal reached
            self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
            path_solution = self.remove_states_behind_goal(self.path_new)
            self.list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=self.dict_node_status,
                                                        list_states_nodes=self.list_status_nodes, time_pause=self.time_pause)
            return True
        return False

    def get_obstacles_information(self):
        """
        Information regarding the obstacles.
        Returns a list of obstacles' information, each element
        contains information regarding an obstacle:
        [x_center_position, y_center_position, length, width]

        """
        return self.extract_collision_obstacles_information()

    def get_goal_information(self):
        """
        Information regarding the goal.
        Returns a list of the goal's information
        with the following form:
        [x_center_position, y_center_position, length, width]
        """
        return self.extract_goal_information()

    def get_node_information(self, node_current):
        """
        Information regarding the input node_current.
        Returns a list of the node's information
        with the following form:
        [x_center_position, y_center_position]
        """
        return node_current.get_position()

    def get_node_path(self, node_current):
        """
        Information regarding the input node_current.
        Returns the path starting from the initial node and ending at node_current.
        """
        return node_current.get_path()

    def print_results(self, results):
        path = results[0]
        visited_nodes = results[1]
        g_cost = results[2]
        f_start = results[3]
        print("Visited Nodes:", end=" "), print(visited_nodes)
        print("Path:", end=" ")
        i = 0
        for node in path:
            print("(", end="")
            print(round(node[0], 2), end=",")
            if i == len(path) - 1:
                print(round(node[1], 2), end=")\n")
            else:
                print(round(node[1], 2), end=") -> ")
            i += 1
        print("Estimated Cost:", end=" "), print(g_cost)
        print("Heuristic Cost (initial node):", end=" "), print(f_start)

    def cost_function(self, node_current):
        """
        Returns g(n) from initial to current node, !only works with cost nodes!
        """
        velocity = node_current.list_paths[-1][-1].velocity

        node_center = self.get_node_information(node_current)
        goal_center = self.get_goal_information()
        distance_x = goal_center[0] - node_center[0]
        distance_y = goal_center[1] - node_center[1]
        length_goal = goal_center[2]
        width_goal = goal_center[3]

        distance = 4.5
        if abs(distance_x) < length_goal / 2 and abs(distance_y) < width_goal / 2:
            prev_x = node_current.list_paths[-2][-1].position[0]
            prev_y = node_current.list_paths[-2][-1].position[1]
            distance = goal_center[0] - length_goal / 2 - prev_x
        cost = node_current.cost + distance

        return cost

    def heuristic_function(self, node_current, distance_type):
        """
        Enter your heuristic function h(x) calculation of distance from node_current to goal
        Returns the distance normalized to be comparable with cost function measurements
        """
        cur_pos = self.get_node_information(node_current)
        goal_area = self.get_goal_information()
        distance = 0
        if distance_type == "euclidean":
            min_distance, closest_edge = self.calc_closest_edge(node_current,0)
            distance = np.sqrt((closest_edge[0]-cur_pos[0])**2 + (closest_edge[1]-cur_pos[1])**2)
        elif distance_type == "manhattan":
            min_distance, closest_edge = self.calc_closest_edge(node_current,1)
            distance = np.abs(cur_pos[0] - closest_edge[0]) + np.abs(cur_pos[1] - closest_edge[1])
        return distance

    def save_goal_edges(self):
        goal_area = self.get_goal_information()
        self.goal_edges = [[goal_area[0] - goal_area[2] / 2, goal_area[1] + goal_area[3] / 2],
                           [goal_area[0] - goal_area[2] / 2, goal_area[1]],
                           [goal_area[0] - goal_area[2] / 2, goal_area[1] - goal_area[3] / 2],
                           [goal_area[0], goal_area[1] + goal_area[3] / 2],
                           [goal_area[0], goal_area[1]],
                           [goal_area[0], goal_area[1] - goal_area[3] / 2],
                           [goal_area[0] + goal_area[2] / 2, goal_area[1] + goal_area[3] / 2],
                           [goal_area[0] + goal_area[2] / 2, goal_area[1]],
                           [goal_area[0] + goal_area[2] / 2, goal_area[1] - goal_area[3] / 2]]

    def calc_closest_edge(self, node_current, distance_type):
        min_distance = float('inf')
        closest_edge = None
        for edge in self.goal_edges:
            curr_dist = self.distance(node_current.get_position(),edge,distance_type)
            if curr_dist<min_distance:
                min_distance = curr_dist
                closest_edge = edge
        return min_distance, closest_edge

    def evaluation_function(self, node_current, distance_type):
        """
        f(x) = g(x) + h(x)
        """
        g = self.cost_function(node_current)
        h = self.heuristic_function(node_current, distance_type)
        f = g + h
        return f

    def execute_search(self, time_pause, distance_type):
        self.node_initial = self.initialize_search(time_pause=time_pause)
        """Enter your code here"""

        # initiate a* search algorithm
        result = self.astar(node_start=self.node_initial, distance_type=distance_type)

        if type(result) == str:
            # path not found
            print(result)
        else:
            # path found
            self.print_results(result)

        return result

    def astar(self, node_start, distance_type):
        self.save_goal_edges()
        # current cost (node_initial) is zero
        g_start = self.cost_function(node_current=node_start)
        # calculate estimation of path cost from start node to goal node
        f_start = self.evaluation_function(node_current=node_start, distance_type=distance_type)

        # initialize opened list (fringe) as a priority queue
        opened_queue = PriorityQueue()
        opened_queue.insert(item=((node_start, None), g_start), priority=f_start)

        # initialize closed dict (explored nodes)
        """ items in this particular dictionary have the following structure
                    dict = {int: ((CostNode, PrimitiveSuccessor), int)}
                   - key ---> f_cost
                   - value is a data union, consists of:
                       a) a tuple containing the current state as CostNode obj 
                          and the SAME state as a motion primitive
                       b) an integer that represents the path's current cost
               """
        closed_dict = {}

        # visited nodes counter
        visited_nodes = 0

        # while fringe contains items ----> available unexplored nodes
        while not opened_queue.empty():
            # pop tuple value from fringe
            node_current = opened_queue.pop()

            # this counts as exploring a node so we increase the counter and update the visuals
            visited_nodes += 1
            self.update_visuals("currently explored", node_current[0][0])

            # calculate f_cost and g_cost
            g_cost = self.cost_function(node_current=node_current[0][0])
            f_cost = self.evaluation_function(node_current=node_current[0][0], distance_type=distance_type)
            print(self.get_node_information(node_current[0][0]), g_cost)

            # goal-testing
            if node_current[0][1] and self.goal_reached(node_current=node_current[0][0], successor=node_current[0][1]):
                # calculate the path
                path = self.get_node_path(node_current=node_current[0][0])
                # set final estimation of cost
                return path, visited_nodes, g_cost, f_start

            # for each motion primitive from the current state, do:
            for primitive_successor in node_current[0][0].get_successors():
                # take the step in order to produce successor CostNode
                collision_flag, child = self.take_step(successor=primitive_successor, node_current=node_current[0][0])

                # check for obstacle collision
                if collision_flag:
                    # if we collided with smth, skip this node
                    continue

                # calculate f & g cost from start to node_current's child
                g_child_cost = self.cost_function(node_current=child)
                f_child_cost = self.evaluation_function(node_current=child, distance_type=distance_type)

                # search both fringe and explored nodes for more expensive paths
                # this is crucial to ensure optimal path when heuristic is admissible
                # but not consistent
                flag_opened = opened_queue.search_queue(node_current=child)
                flag_closed = search_dict(dictionary=closed_dict, node_current=child)

                if flag_opened != -1:
                    if flag_opened[1] < g_child_cost:
                        continue
                elif flag_closed != -1:
                    if closed_dict[flag_closed][1] < g_child_cost:
                        continue
                    node_data = closed_dict.pop(flag_closed)
                    opened_queue.insert(item=((node_data[0][0], primitive_successor), g_child_cost),priority=f_child_cost)
                else:
                    # add the child to fringe
                    opened_queue.insert(item=((child, primitive_successor), g_child_cost),priority=f_child_cost)
                self.update_visuals("in fringe", child)

            # add node_current to explored nodes dict
            closed_dict[f_cost] = ((node_current[0][0], node_current[0][1]), g_cost)
            self.update_visuals("explored", node_current[0][0])
        # end of while loop - rinse and repeat :)

    # if pc reaches this point it means that the fringe is empty,
    # and by extension our search yielded no results
        return "Path Not Found"


class Astar(SequentialSearch):
    """
    Class for Astar Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
 
