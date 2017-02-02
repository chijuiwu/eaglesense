
/**
DEPRECATED

EagleSense skeleton geodesic

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"

#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/math/special_functions/round.hpp>

#include <math.h>
#include <numeric>
#include <iostream>

#include "topviewkinect/skeleton/geodesic.h"

namespace topviewkinect
{
    namespace skeleton
    {
        bool operator== (const DepthVertex &lhs, const DepthVertex &rhs)
        {
            return (lhs.x == rhs.x && lhs.y == rhs.y);
        }

        size_t DepthVertexHash::operator()(const DepthVertex& vertex) const
        {
            int hash = 23;
            hash = hash * 31 + vertex.x;
            hash = hash * 31 + vertex.y;
            return hash;
        }

        bool operator== (const DepthVertexHash &lhs, const DepthVertexHash &rhs)
        {
            return (lhs == rhs);
        }

        static bool compare_geodesic_distance(const std::pair<DepthVertex, double>& lhs, const std::pair<DepthVertex, double>& rhs)
        {
            // Max distance = no edge from or to the vertex
            if (lhs.second == DBL_MAX)
            {
                return true;
            }
            else if (rhs.second == DBL_MAX)
            {
                return false;
            }
            else
            {
                return lhs.second < rhs.second;
            }
        }

        void DepthGeodesicGraph::calc_geodesic_frame(const cv::Mat& src, cv::Mat& dst, const int downsampled_x_scale, const int downsampled_y_scale, const DepthVertex& root)
        {
            DepthGeodesicGraph graph;

            // Create the geodeisc graph, iterate the source image from left to right, then connect the top-left, top-middle, top-right, and right vertices (the graph is undirected)
            const int src_width = src.size().width;
            const int src_height = src.size().height;
            const unsigned char* p_src_data = src.data;
            for (int row = 0; row < src.size().height; row += downsampled_y_scale)
            {
                for (int col = 0; col < src.size().width; col += downsampled_x_scale)
                {
                    DepthVertex current{ col, row };
                    unsigned char current_depth = p_src_data[src_width * row + col];
                    if (current_depth == 0)
                    {
                        continue;
                    }

                    // top
                    if (row > downsampled_y_scale - 1)
                    {
                        DepthVertex top{ col, row - downsampled_y_scale };
                        unsigned char top_depth = p_src_data[src_width * (row - downsampled_y_scale) + col];
                        if (top_depth > 0)
                        {
                            graph.add_edge(current, current_depth, top, top_depth);
                        }
                        // top-left
                        if (col > downsampled_x_scale - 1)
                        {
                            DepthVertex top_left{ col - downsampled_x_scale, row - downsampled_y_scale };
                            unsigned char top_left_depth = p_src_data[src_width * (row - downsampled_y_scale) + (col - downsampled_x_scale)];
                            if (top_left_depth > 0)
                            {
                                graph.add_edge(current, current_depth, top_left, top_left_depth);
                            }
                        }
                        // top-right
                        if (col < src_width - downsampled_x_scale)
                        {
                            DepthVertex top_right{ col + downsampled_x_scale, row - downsampled_y_scale };
                            unsigned char top_right_depth = p_src_data[src_width * (row - downsampled_y_scale) + (col + downsampled_x_scale)];
                            if (top_right_depth > 0)
                            {
                                graph.add_edge(current, current_depth, top_right, top_right_depth);
                            }
                        }
                    }

                    // right
                    if (col < src_width - downsampled_x_scale)
                    {
                        DepthVertex right{ col + downsampled_x_scale, row };
                        unsigned char right_depth = p_src_data[src_width * row + (col + downsampled_x_scale)];
                        if (right_depth > 0)
                        {
                            graph.add_edge(current, current_depth, right, right_depth);
                        }
                    }
                }
            }

            // Run the shortest path algorithm from the root node
            std::unordered_map<DepthVertex, double, DepthVertexHash> shortest_geodesic_paths = graph.find_shortest_paths(root);

            // Create the image (normalized geodesic distances)
            dst = cv::Mat::zeros(src.size(), CV_8UC1);
            double max_distance = std::max_element(shortest_geodesic_paths.begin(), shortest_geodesic_paths.end(), compare_geodesic_distance)->second;
            for (auto const& vertex_distance : shortest_geodesic_paths)
            {
                int frame_x = vertex_distance.first.x;
                int frame_y = vertex_distance.first.y;
                if (frame_x < 0 || frame_y < 0)
                {
                    continue;
                }
                double distance = vertex_distance.second;
                int color = (distance == DBL_MAX) ? 255 : boost::math::iround(distance * 255 / max_distance);
                cv::Rect roi(frame_x, frame_y, downsampled_x_scale, downsampled_y_scale);
                dst(roi).setTo(color);
            }
        }

        size_t DepthGeodesicGraph::num_vertices() const
        {
            return boost::num_vertices(this->graph);
        }

        size_t DepthGeodesicGraph::num_edges() const
        {
            return boost::num_edges(this->graph);
        }

        void DepthGeodesicGraph::add_edge(const DepthVertex& from, const unsigned char from_depth, const DepthVertex& to, const unsigned char to_depth)
        {
            if (this->vertices.count(from) == 0)
            {
                this->vertices[from] = boost::add_vertex(graph);
            }

            if (this->vertices.count(to) == 0)
            {
                this->vertices[to] = boost::add_vertex(graph);
            }

            int edge_vector[3] = { to.x - from.x, to.y - from.y, to_depth - from_depth };
            double weight = std::sqrt(std::inner_product(std::begin(edge_vector), std::end(edge_vector), std::begin(edge_vector), 0.0));
            DepthEdgeWeightProperty edge_weight_ab(weight);
            depth_vertex_t vertex_a = vertices[from];
            depth_vertex_t vertex_b = vertices[to];
            boost::add_edge(vertex_a, vertex_b, edge_weight_ab, this->graph);
        }

        std::unordered_map<DepthVertex, double, DepthVertexHash> DepthGeodesicGraph::find_shortest_paths(const DepthVertex& root)
        {
            std::vector<depth_vertex_t> parents(boost::num_vertices(this->graph));
            std::vector<double> distances(boost::num_vertices(this->graph));
            boost::dijkstra_shortest_paths(this->graph, this->vertices[root], boost::predecessor_map(&parents[0]).distance_map(&distances[0]));

            std::unordered_map<DepthVertex, double, DepthVertexHash> shortest_paths;
            for (const auto& kv : this->vertices)
            {
                shortest_paths[kv.first] = distances[kv.second];
            }
            return shortest_paths;
        }

        const DepthVertex DepthGeodesicGraph::get_vertex(const int x, const int y) const
        {
            if (this->vertices.find(DepthVertex{ x, y }) != this->vertices.end()) {
                return DepthVertex{ x, y };
            }

            double nearest_distance = DBL_MAX;
            int nearest_x = -1;
            int nearest_y = -1;
            for (auto const& entry : this->vertices)
            {
                int dx = entry.first.x - x;
                int dy = entry.first.y - y;
                double distance = std::sqrt(dx * dx + dy * dy);
                if (distance < nearest_distance)
                {
                    nearest_distance = distance;
                    nearest_x = entry.first.x;
                    nearest_y = entry.first.y;
                }
            }

            return DepthVertex{ nearest_x, nearest_y };
        }
    }
}
