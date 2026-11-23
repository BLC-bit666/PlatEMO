"""Exact route evaluation with bounded decision/path/result caches.

One instance belongs to one sequential process. Targeted exact search and original track
statistics preserve outputs; parallel-edge ties keep the original row order.
"""
from collections import OrderedDict
from itertools import product
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from native_kernels import kernels

from route_problem import (BlackForestRouteProblem, RouteGraph,
                           RouteNode, RoutePath, shortest_path, build_track)
from bike_router.core.route_path import RouteEdge
from bike_router.core.graph_store import _oriented_geometry, str_or_none
from bike_router.core.constants import CostConfig, GpxConfig, Mode, RailConfig
from bike_router.core.cost import road_weight, surface_weight


class FastBlackForestRouteProblem(BlackForestRouteProblem):
    def __init__(self, cache_size=1024):
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError('cache_size must be a nonnegative integer')
        super().__init__()
        self.cache_size = cache_size
        self._decisions = OrderedDict()
        self._paths = OrderedDict()
        self._routes = OrderedDict()
        self._nodes = {}
        self._edges = {}
        self._stats = dict(requests=0, exact_decision_hits=0, uncached_decisions=0,
                           shortest_path_calls=0, full_searches=0, bounded_searches=0,
                           bounded_fallbacks=0, route_result_hits=0, route_result_builds=0,
                           cost_component_updates=0, astar_searches=0, astar_visited_nodes=0)
        self._last_edges = None
        self._kernels = kernels()
        edges = self.edges
        self._length = edges.length_m.to_numpy(dtype=np.float64)
        self._length_km = self._length / GpxConfig.METERS_PER_KM
        from_elev = edges.from_node.map(self.elevation).to_numpy(dtype=np.float64)
        to_elev = edges.to_node.map(self.elevation).to_numpy(dtype=np.float64)
        self._climb_ratio = np.maximum(to_elev-from_elev, 0.0) / CostConfig.UPHILL_REFERENCE_M
        self._surface = edges.surface.map(surface_weight).to_numpy(dtype=np.float64)
        self._road = edges.highway.map(road_weight).to_numpy(dtype=np.float64)
        mode = edges['mode'].to_numpy()
        self._masks = [mode == Mode.BIKE, mode == Mode.RAIL, mode == Mode.STATION]
        self._mode = np.zeros(len(edges), dtype=np.uint8)
        self._mode[self._masks[1]] = 1
        self._mode[self._masks[2]] = 2
        assert np.all(self._length[self._masks[2]] <= RailConfig.STATION_RADIUS_M)
        assert np.all(self._length >= 0) and np.all(np.logical_or.reduce(self._masks))
        self._graph = RouteGraph.from_arrays(**self.arrays, cost=self._length)
        u = np.asarray([self._graph.index[int(v)] for v in self.arrays['from_osmid']])
        v = np.asarray([self._graph.index[int(v)] for v in self.arrays['to_osmid']])
        self._order = np.lexsort((v, u))
        us, vs = u[self._order], v[self._order]
        first = np.r_[True, (us[1:] != us[:-1]) | (vs[1:] != vs[:-1])]
        self._starts = np.flatnonzero(first)
        self._ends = np.r_[self._starts[1:], len(self._order)]
        self._first_rows = self._order[self._starts]
        self._extra_rows = self._order[~first]
        self._extra_groups = np.repeat(np.arange(len(self._starts)), self._ends-self._starts-1)
        assert np.all(np.diff(self._order)[~first[1:]] > 0)
        assert len(self._starts) == self._graph.matrix.nnz
        assert np.array_equal(self._graph.matrix.indices, vs[first])
        assert np.array_equal(np.repeat(np.arange(len(self.nodes)), np.diff(self._graph.matrix.indptr)), us[first])
        matrix = self._graph.matrix
        self._csr_rows = np.repeat(np.arange(len(self.nodes)), np.diff(matrix.indptr))
        self._incoming = np.argsort(matrix.indices, kind='stable')
        self._incoming_ptr = np.r_[0, np.cumsum(np.bincount(matrix.indices, minlength=len(self.nodes)))]
        self._elevations = self.nodes.elevation_m.to_numpy()
        self._stations = self.nodes.station_name.to_numpy()
        self._edge_columns = {name: edges[name].to_numpy() for name in
                              ['from_node', 'to_node', 'mode', 'surface', 'highway', 'geometry_wkt']}
        self._cost_buffer = np.empty(len(edges), dtype=np.float64)
        for values in [self._length, self._length_km, self._climb_ratio, self._surface, self._road]:
            assert np.isfinite(values).all() and (values >= 0).all()
        # All five coefficients lie in [0,100] and all features are nonnegative.
        # Checking the upper corner once proves finite nonnegative edge costs.
        assert np.isfinite(self._edge_costs(np.full(5, 100.))).all()
        self._corners = np.asarray(list(product([0., 100.], repeat=5)))
        bounds = []
        target = self._graph.index[self.end]
        for x in self._corners:
            self.routing_graph(self._edge_costs(x))
            bounds.append(dijkstra(self._graph.matrix.transpose().tocsr(), directed=True, indices=target))
        self._bounds = np.asarray(bounds)
        self._distance = np.empty(len(self.nodes), dtype=np.float64)
        self._predecessors = np.empty(len(self.nodes), dtype=np.int32)
        self._stats['cost_component_updates'] = 0

    def _remember(self, cache, key, value):
        if self.cache_size:
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
        return value

    def statistics(self):
        return dict(self._stats, decision_cache_entries=len(self._decisions),
                    route_cache_entries=len(self._routes), path_cache_entries=len(self._paths),
                    cached_nodes=len(self._nodes), cached_edges=len(self._edges), cache_limit=self.cache_size)

    def _node(self, osmid):
        node = self._nodes.get(osmid)
        if node is None:
            i = self._graph.index[osmid]
            station = self._stations[i]
            node = RouteNode(int(osmid), float(self._graph.lat[i]), float(self._graph.lon[i]),
                             float(self._elevations[i]), str(self._graph.node_type[i]),
                             None if pd.isna(station) else str(station))
            self._nodes[osmid] = node
        return node

    def _edge(self, row):
        edge = self._edges.get(row)
        if edge is None:
            cols = self._edge_columns
            source = int(cols['from_node'][row]); target = int(cols['to_node'][row])
            geometry, geometry_z = _oriented_geometry(wkt=cols['geometry_wkt'][row], node_a=self._node(source))
            edge = RouteEdge(from_node=source, to_node=target, mode=str(cols['mode'][row]),
                             length_m=float(self._length[row]), surface=str_or_none(value=cols['surface'][row]),
                             highway=str_or_none(value=cols['highway'][row]), geometry=geometry, geometry_z=geometry_z)
            self._edges[row] = edge
        return edge

    def _selected_edges(self, path, costs):
        groups = self._paths.get(path)
        if groups is None:
            graph = self._graph; matrix = graph.matrix
            nodes = [graph.index[node] for node in path]
            groups = []
            for source, target in zip(nodes[:-1], nodes[1:]):
                lo, hi = matrix.indptr[source:source+2]
                group = int(lo + np.searchsorted(matrix.indices[lo:hi], target))
                assert group < hi and matrix.indices[group] == target
                groups.append(group)
            groups = self._remember(self._paths, path, np.asarray(groups, dtype=np.int64))
        else:
            self._paths.move_to_end(path)
        selected = self._order[self._starts[groups]].copy()
        for i in np.flatnonzero(self._ends[groups] - self._starts[groups] > 1):
            group = groups[i]
            candidates = self._order[self._starts[group]:self._ends[group]]
            selected[i] = candidates[np.argmin(costs[candidates])]
        return tuple(int(row) for row in selected)

    def _full_path(self):
        self._stats['full_searches'] += 1
        self._stats['shortest_path_calls'] += 1
        return tuple(shortest_path(route_graph=self._graph, source_osmid=self.start, target_osmid=self.end))

    def _path_is_unique(self, distance, pred, path):
        # Guard against ties AND accumulated floating-point rounding. An
        # ambiguous predecessor anywhere on the target path uses the original
        # unbounded kernel, preserving its heap-dependent canonical choice.
        margin = 8*len(self._graph.osmids)*np.spacing(max(float(distance[path[-1]]), 1.0))
        for v in path[1:]:
            edges = self._incoming[self._incoming_ptr[v]:self._incoming_ptr[v+1]]
            u = self._csr_rows[edges]
            alternative = (u != pred[v]) & (distance[u]+self._graph.matrix.data[edges] <= distance[v]+margin)
            if np.any(alternative):
                return False
        return True

    def _find_path(self, costs):
        graph = self._graph; source = graph.index[self.start]; target = graph.index[self.end]
        matrix = graph.matrix
        # Each fixed corner gives a reverse shortest-distance lower bound.
        # f=min(x_i/100 for its nonzero coefficients, 1) implies cost(x)>=f*cost(corner).
        # Choose the strongest bound at this source; the kernel discounts it
        # conservatively for positive-sum floating-point rounding.
        factors = np.min(np.where(self._corners > 0, self._current_x/100., 1.), axis=1)
        selected = int(np.argmax(factors*self._bounds[:, source]))
        visited = self._kernels.target_astar(len(self.nodes), matrix.indptr, matrix.indices, matrix.data,
            source, target, self._bounds[selected], float(factors[selected]), self._distance, self._predecessors)
        self._stats['shortest_path_calls'] += 1
        self._stats['astar_searches'] += 1
        self._stats['astar_visited_nodes'] += visited
        distance, pred = self._distance, self._predecessors
        if np.isfinite(distance[target]):
            path = []; cursor = target
            while cursor != source:
                path.append(cursor); cursor = int(pred[cursor]); assert cursor >= 0
            path.append(source); path.reverse()
            if self._path_is_unique(distance, pred, path):
                return tuple(int(graph.osmids[i]) for i in path)
        # Ambiguous equal/near-equal paths retain the upstream heap's canonical choice.
        self._stats['bounded_fallbacks'] += 1
        return self._full_path()

    def costs(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (5,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 100)):
            raise ValueError('Decision domain is [0,100]^5')
        return self._edge_costs(x).copy()

    def _edge_costs(self, x):
        # One native pass; keep the original multiply/add order and rounding.
        x = np.ascontiguousarray(x, dtype=np.float64)
        self._kernels.route_costs(len(self.edges), x, self._length, self._length_km,
            self._climb_ratio, self._surface, self._road, self._mode, self._cost_buffer)
        self._stats['cost_component_updates'] += 5
        return self._cost_buffer

    def routing_graph(self, costs):
        # Same min cost per directed (u,v), same sorted CSR indices as upstream.
        data = self._graph.matrix.data
        data[:] = costs[self._first_rows]
        np.minimum.at(data, self._extra_groups, costs[self._extra_rows])
        return self._graph

    def evaluate(self, x, save_route=None):
        x = np.asarray(x, dtype=float)
        if x.shape != (5,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 100)):
            raise ValueError('Decision domain is [0,100]^5')
        start = time.perf_counter()
        self._stats['requests'] += 1
        key = x.tobytes()
        payload = self._decisions.get(key)
        if payload is not None:
            self._stats['exact_decision_hits'] += 1
            self._decisions.move_to_end(key)
        else:
            self._stats['uncached_decisions'] += 1
            self._current_x = x
            costs = self._edge_costs(x)
            self.routing_graph(costs)
            path_ids = self._find_path(costs)
            edge_ids = self._selected_edges(path_ids, costs)
            payload = self._routes.get(edge_ids)
            if payload is not None:
                self._stats['route_result_hits'] += 1
                self._routes.move_to_end(edge_ids)
            else:
                path = RoutePath([self._node(node) for node in path_ids], [self._edge(row) for row in edge_ids])
                track = build_track(route=path)
                self._stats['route_result_builds'] += 1
                payload = dict(path_ids=path_ids, edge_ids=edge_ids,
                               objectives=(track.total.distance_km, track.bike.ascent_m),
                               constraint=int(any(edge.mode != 'bike' for edge in path.edges)),
                               modes=tuple(sorted({edge.mode for edge in path.edges})),
                               bike_km=track.bike.distance_km,
                               rail_km=sum(edge.length_m for edge in path.edges if edge.mode == 'rail')/1000)
                assert np.isfinite(payload['objectives']).all()
                self._remember(self._routes, edge_ids, payload)
            self._remember(self._decisions, key, payload)
        self._last_edges = payload['edge_ids']
        elapsed = time.perf_counter()-start
        if save_route:
            Path(save_route).write_text(json.dumps({'nodes': [self._node(n).__dict__ for n in payload['path_ids']],
                                                   'edges': [self._edge(e).__dict__ for e in payload['edge_ids']]}, indent=2))
        return dict(x=x.tolist(), objectives=list(payload['objectives']), constraint=payload['constraint'],
                    edge_count=len(payload['edge_ids']), modes=list(payload['modes']),
                    bike_km=payload['bike_km'], rail_km=payload['rail_km'], seconds=elapsed)
