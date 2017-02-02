
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

#pragma once

#include <opencv2/core.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>

#include <unordered_map>

namespace topviewkinect
{
    namespace skeleton
    {
        struct DepthVertex
        {
            int x;
            int y;
        };

        bool operator== (const DepthVertex& lhs, const DepthVertex& rhs);

        struct DepthVertexHash
        {
            size_t operator() (const DepthVertex& vertex) const;
        };

        bool operator== (const DepthVertexHash &lhs, const DepthVertexHash &rhs);

        typedef boost::property<boost::edge_weight_t, double> DepthEdgeWeightProperty;
        typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, DepthVertex, DepthEdgeWeightProperty> UndirectedDepthGraph;
        typedef boost::graph_traits<UndirectedDepthGraph>::vertex_descriptor depth_vertex_t;
        typedef boost::graph_traits<UndirectedDepthGraph>::edge_descriptor depth_edge_t;

        class DepthGeodesicGraph
        {
        private:
            std::unordered_map<DepthVertex, depth_vertex_t, DepthVertexHash> vertices;
            UndirectedDepthGraph graph;

        public:
            DepthGeodesicGraph() {}
            ~DepthGeodesicGraph() {}

            static void calc_geodesic_frame(const cv::Mat& src, cv::Mat& dst, const int subsampled_x_scale, const int subsampled_y_scale, const DepthVertex& root);

            size_t num_vertices() const;
            size_t num_edges() const;

            void add_edge(const DepthVertex& from, const unsigned char from_depth, const DepthVertex& to, const unsigned char to_depth);
            std::unordered_map<DepthVertex, double, DepthVertexHash> find_shortest_paths(const DepthVertex& root);
            const DepthVertex get_vertex(const int x, const int y) const;
        };
    }
}
