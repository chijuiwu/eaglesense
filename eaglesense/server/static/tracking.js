var m_eaglesense;

// var last_timestamp = 0;
// var skeleton_path = [];
//var skeleton_positions = [{"x":10,"y":10},{"x":20,"y":20}];
//var skeleton_path = [{"x1":10,"y1":10,"x2":20,"y2":20,"rgb":d3.rgb(49, 130, 189),"width":3,"a":0.5},{"x1":20,"y1":20,"x2":30,"y2":30,"rgb":d3.rgb(49, 130, 189),"width":3,"a":0.5}];

// heatmap
var tracking_svg;
var margin, width, height;
var skeleton_positions = [];
// var randomX = d3.random.normal(width / 2, 80),
//     randomY = d3.random.normal(height / 2, 80),
//     points = d3.range(2000).map(function() { return [randomX(), randomY()]; });
var hexbin;
var radius = 50;
var x,y,xAxis,yAxis;
var min_length = 0;
var max_length = 0;
var max_framesize = 1000;

function setup_tracking(tracking_div, width, height) {
    m_eaglesense = new EagleSenseWebSocket();

    margin = {top: 0, right: 0, bottom: 0, left: 0},
    width = width - margin.left - margin.right,
    height = height - margin.top - margin.bottom;

    hexbin = d3.hexbin()
        .size([width, height])
        .radius(radius);

    x = d3.scale.identity()
        .domain([0, width]);

    y = d3.scale.linear()
        .domain([0, height])
        // .range([height, 0]);

    xAxis = d3.svg.axis()
        .scale(x)
        .orient("top")
        // .tickSize(6, -height);

    yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        // .tickSize(6, -width);

    tracking_svg = d3.select(tracking_div).append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    tracking_svg.append("clipPath")
        .attr("id", "clip")
      .append("rect")
        .attr("class", "mesh")
        .attr("width", width)
        .attr("height", height);

    // svg.append("g")
    //     .attr("class", "y axis")
    //     .call(yAxis);

    // svg.append("g")
    //     .attr("class", "x axis")
    //     .attr("transform", "translate(0," + height + ")")
    //     .call(xAxis);

}

function start_tracking() {
    var onopnen_callback = on_server_connected;
    var onmessage_callback = on_skeleton_data;
    var onclose_callback = on_server_disconnected;
    var onerror_callback = on_server_disconnected;

    m_eaglesense.connect(
        onopnen_callback, onmessage_callback, onclose_callback, onerror_callback
    );
}

function on_server_connected() {
    $("#server_online").removeClass("hidden");
    $("#server_offline").addClass("hidden");
    $("#kinect_online").removeClass("hidden");
    $("#kinect_offline").addClass("hidden");
}

function on_server_disconnected() {
    $("#server_online").addClass("hidden");
    $("#server_offline").removeClass("hidden");
    $("#kinect_online").addClass("hidden");
    $("#kinect_offline").removeClass("hidden");

    // clear visualization
    d3.selectAll("circle").remove();
    d3.selectAll("line").remove();
}

var steps = 0;

function on_skeleton_data(data) {
    skeletons = data["skeletons"];

    // update people count
    num_people = skeletons.length;
    $("#num_people").html(num_people);

    // update phones and tablets count
    num_phones = 0;
    num_tablets = 0;
    for (var i = 0; i < skeletons.length; i++) {
        skeleton_activity = skeletons[i]["activity"].toLowerCase();
        if (skeleton_activity === "phone") {
            num_phones += 1;
        }
        else if (skeleton_activity === "tablet") {
            num_tablets += 1;
        }
    }
    $("#num_phones").html(num_phones);
    $("#num_tablets").html(num_tablets);

    // timestamp = result_data["timestamp"];
    // if (timestamp - last_timestamp < 1000) {
    //     return;
    // }
    // else {
    //     last_timestamp = timestamp;
    // }

    // if (skeleton_positions.length == 101) {
    //     skeleton_positions.shift();
    // }

    for (var i = 0; i < skeletons.length; i++) {
        skeleton = skeletons[i];
        if (skeleton["activity_tracked"] == 0) {
            continue;
        }
        skeleton_positions.push([skeleton["head"]["x"] - 10, skeleton["head"]["y"] - 10]);
        steps++;
        // if (skeleton_positions.length > 1) {
        //     previous_pos = skeleton_positions[skeleton_positions.length - 2];
        //     skeleton_path.push({
        //         "x1": previous_pos["head"]["x"],
        //         "y1": previous_pos["head"]["y"],
        //         "x2": skeleton["head"]["x"],
        //         "y2": skeleton["head"]["y"],
        //         "rgb": d3.rgb(49, 130, 189),
        //         "width": 5
        //     });
        // }
    }

    if (steps > max_framesize) {
        skeleton_positions.shift();
        steps = max_framesize;
    }
    // else {
    //     skeleton_positions = [];
    //     skeleton_path = [];
    // }

    // // update color
    // for (var i = 0; i < skeleton_path.length; i++) {
    //     skeleton_path[i]["a"] = i / 100;
    // }

    update();
}

function update() {
    var min_length = Number.MAX_VALUE;
    var max_length = Number.MIN_VALUE;
    var hexbin_points = hexbin(skeleton_positions);
    for (var i = 0; i < hexbin_points.length; i++) {
        var hexbin_point_length = hexbin_points[i].length;
        if (hexbin_point_length < min_length) {
            min_length = hexbin_point_length;
        }
        if (hexbin_point_length > max_length) {
            max_length = hexbin_point_length;
        }
    }

    var range = (max_length - min_length) / 8;

    var color = d3.scale.linear()
        .domain([0, range * 1, range * 2, range * 3, range * 4, range * 5, range * 6, range * 7, range * 8])
        .range(["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c"]);

    tracking_svg.append("g")
        .attr("clip-path", "url(#clip)")
      .selectAll(".hexagon")
        .data(hexbin_points)
      .enter().append("path")
        .attr("class", "hexagon")
        .attr("d", hexbin.hexagon())
        .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
        .style("fill", function(d) { return color(d.length); });

    // d3.selectAll("line").remove();

    // var lines = tracking_svg.selectAll("line").data(skeleton_path);
    // lines.enter()
    //     .append("line")
    //     .attr("x1", function(d) { return d["x1"] } )
    //     .attr("y1", function(d) { return d["y1"] } )
    //     .attr("x2", function(d) { return d["x2"] } )
    //     .attr("y2", function(d) { return d["y2"] } )
    //     .attr("stroke", function(d) { return d["rgb"] })
    //     .attr("fill", function(d) { return d["rgb"] })
    //     .attr("opacity", function(d) { return d["a"] } )
    //     .attr("stroke-width", function(d) { return d["width"] } )
}
