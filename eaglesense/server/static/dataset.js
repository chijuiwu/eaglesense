var orientation_svg;

function setup_orientation_visualization(orientation_visualization_div, width, height) {
    orientation_svg = d3.select(orientation_visualization_div).append("svg")
            .attr("width", width)
            .attr("height", height);

    orientation_svg.append("rect")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("fill", "#000000");

    // arrow marker
    defs = orientation_svg.append("defs")
    defs.append("marker")
            .attr({
                "id":"arrow",
                "viewBox":"0 -5 10 10",
                "refX":5,
                "refY":0,
                "markerWidth":4,
                "markerHeight":4,
                "orient":"auto"
            })
            .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("class","arrowHead")
                .style("fill", "grey");

//    .append("svg:defs").append("svg:marker")
//                   .attr("id", "triangle")
//                   .attr("refX", 6)
//                   .attr("refY", 6)
//                   .attr("markerWidth", 30)
//                   .attr("markerHeight", 30)
//                   .attr("orient", "auto")
//                   .append("path")
//                   .attr("d", "M0,-5L10,0L0,5")
//                   .style("fill", "black")
//                   .attr("class","arrowHead");
}

function update_orientation_visualization(orientation) {

    orientation_svg.selectAll("circle").remove();
    orientation_svg.selectAll("line").remove();

    if (orientation === undefined) {
        console.log("undefined orientation")
        return;
    }

//    var circles_json = [
//        { "x_axis": 50, "y_axis": 50, "radius": 8, "color" : "red" }
//    ];
//
//    var circles = orientation_svg.selectAll("circle")
//                                 .data(circles_json)
//                                 .enter()
//                                 .append("circle");
//
//    var circles_attributes = circles
//                             .attr("cx", function (d) { return d.x_axis; })
//                             .attr("cy", function (d) { return d.y_axis; })
//                             .attr("r", function (d) { return d.radius; })
//                             .style("fill", function(d) { return d.color; });

    orientation_svg.append("circle")
        .attr({
            cx:25, cy:25, r:3, fill: "grey"
        })

    x2 = Math.round(25 + 15 * Math.cos(orientation * Math.PI / 180));
    y2 = Math.round(25 + 15 * Math.sin(orientation * Math.PI / 180));

    orientation_svg.append("line")
       .attr({
//           "class":"arrow",
           "marker-end": "url(#arrow)",
           "x1": 25,
           "y1": 25,
           "x2": x2,
           "y2": y2,
           "stroke-width": 2,
           "stroke": "grey"
       });

}