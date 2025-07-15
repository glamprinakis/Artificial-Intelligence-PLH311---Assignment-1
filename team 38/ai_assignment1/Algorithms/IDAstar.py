from SMP.motion_planner.node import Node, CostNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization
import copy
import sys
from abc import ABC
from typing import Tuple, Union, Dict
import numpy as np

sys.path.append('../')

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
                                        list_primitives=[self.motion_primitive_initial],
                                        depth_tree=0, cost=0)
        else:
            node_initial = Node(list_paths=[[self.state_initial]],
                                list_primitives=[self.motion_primitive_initial],
                                depth_tree=0)
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
            return True, child
        return False, child

    def update_visuals(self, node_current, frontier_flag):
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

    def search_list(self, lis, node):
        for n in lis:
            if n[0] == node:
                return True
        return False

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
        if abs(distance_x) < length_goal/2 and abs(distance_y) < width_goal/2:
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
            distance = np.sqrt(np.abs(cur_pos[0] - (goal_area[0] - goal_area[2]/2))**2 + (np.abs(cur_pos[1] - goal_area[1]))**2)
        elif distance_type == "manhattan":
            distance = np.abs(cur_pos[0] - goal_area[0] - goal_area[2] / 2) + np.abs(cur_pos[1] - goal_area[1])
        return distance

    def evaluation_function(self, node_current, distance_type):
        """
        f(x) = g(x) + h(x)
        """
        g = self.cost_function(node_current)
        h = self.heuristic_function(node_current, distance_type)
        f = g + h
        return f

    def print_results(self, path, f_start):
        print("Visited Nodes:", end=" "), print(self.visited_nodes_counter)
        print("Path:", end=" ")
        i = 0
        node_last = None
        for node in path:
            print("(", end="")
            print(round(self.get_node_information(node[0])[0], 2), end=",")
            if i == len(path) - 1:
                print(round(self.get_node_information(node[0])[1], 2), end=")\n")
                node_last = node[0]
            else:
                print(round(self.get_node_information(node[0])[1], 2), end=") -> ")
            i += 1
        print("Estimated Cost:", end=" "), print(self.cost_function(node_last))
        print("Heuristic Cost (initial node):", end=" "), print(f_start)

    def execute_search(self, time_pause, distance_type):
        node_initial = self.initialize_search(time_pause=time_pause)
        """Enter your code here"""
        initial_bound = self.evaluation_function(node_current=node_initial, distance_type=distance_type)
        bound = initial_bound
        path_current = [(node_initial, None)]
        self.visited_nodes_counter = 0

        while True:
            t = self.ida_star(path_current, bound, node_initial=node_initial, distance_type=distance_type)
            if t == "Found":
                self.print_results(path_current, initial_bound)
                return path_current, bound
            if t == float('inf'):
                return "Not Found"
            bound = t

    def ida_star(self, path, bound, node_initial, distance_type):
        # pop last node from stack
        node_current = path[-1][0]
        # this is visiting the node, hence we increase the counter
        self.visited_nodes_counter += 1
        self.update_visuals(node_current, "currently explored")
        # calculate paid cost (node_start...node_current) + estimated cost (node_current...goal)
        f = self.evaluation_function(node_current, distance_type=distance_type)

        # if current_cost > bound
        if self.cost_function(node_current) > bound:
            # we reached the search depth bound
            # set new bound by returning the calculated bound
            self.update_visuals(node_current, "explored")
            return f

        # if current node is the initial node, skip goal testing
        if node_current == node_initial:
            pass
        else:
            # otherwise point to current CostNode and its motion primitive counterpart
            node_current = path[-1][0]
            node_as_successor = path[-1][1]
            # goal testing NEEDS to occur here in order for IDA* to work as intended
            if self.goal_reached(successor=node_as_successor, node_current=node_current):
                # if goal found stop search
                return "Found"

        minimum = float('inf')

        # start iterating through current nodes children
        for primitive_successor in node_current.get_successors():
            # take step to produce CostNode
            collision_flag, child = self.take_step(successor=primitive_successor, node_current=node_current)
            # if collision was detected, skip the current successor
            if collision_flag:
                self.update_visuals(node_current, "explored")
                continue
            # if the child node is not in current path (avoid circling around)
            if not self.search_list(path, child):
                # add aformentioned child CostNode and its motion primitive counterpart to current path
                path.append((child, primitive_successor))
                self.update_visuals(child, "in fringe")
                # perform recursive search
                t = self.ida_star(path, bound, node_initial=node_initial, distance_type=distance_type)
                if t == "Found":
                    # goal was found in nested call, thus we end the recursion
                    return "Found"
                if t < minimum:
                    minimum = t
                    self.update_visuals(child, "explored")
                path.pop()
                self.update_visuals(node_current, "explored")

        return minimum


class IterativeDeepeningAstar(SequentialSearch):
    """
    Class for Iterative Deepening Astar Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
