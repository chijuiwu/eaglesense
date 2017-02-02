var m_eaglesense;

var opt_in_out_svg;
var opt_in_out_width;
var opt_in_out_height;

var stage_1 = true;
var stage_2 = false;
var stage_3 = false;
var paused = false;
var exiting = false;
var exiting_timestamp = 0;

var ship_x = 200;
var ship_y = 500;
var ship_width = 10;
var asteroids = [];
var bullets = [];
var score = 0;
var ship_points = [];
var ship_damages = [];

function setup_opt_in_out(opt_in_out_div, width, height) {
    m_eaglesense = new EagleSenseWebSocket();

    opt_in_out_width = width;
    opt_in_out_height = height;
    opt_in_out_svg = d3.select(opt_in_out_div).append("svg")
        .attr("width", width)
        .attr("height", height);
    opt_in_out_svg.append("rect")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("fill", "#000000");

    ship_x = opt_in_out_width / 2;
    ship_y = opt_in_out_height - 100;

    draw();

    // testing
//    window.requestAnimationFrame(loop);
}

function start_opt_in_out() {
    var onopnen_callback = on_server_connected;
    var onmessage_callback = on_skeleton_data;
    var onclose_callback = on_server_disconnected;
    var onerror_callback = on_server_disconnected;

    m_eaglesense.connect(
        onopnen_callback, onmessage_callback, onclose_callback, onerror_callback
    );
}


function on_server_connected() {
}

function on_server_disconnected() {
}

function on_skeleton_data(data) {
    // assume one person
    var skeletons = data["skeletons"];
    var num_people = skeletons.length;

    if (stage_1 && num_people > 0) {
        stage_1 = false;
        stage_2 = true;
        stage_3 = false;
        paused = false;
    }

    if (stage_2 && num_people > 0) {
        if (skeletons[0]["activity"].toLowerCase() === "pointing") {
            stage_1 = false;
            stage_2 = false;
            stage_3 = true;
            paused = false;
            window.requestAnimationFrame(loop);
        }
    }

    if (stage_3 && num_people > 0) {
        exiting = false;

        skeleton_activity = skeletons[0]["activity"].toLowerCase();
        skeleton_head = skeletons[0]["head"];

        if (skeleton_head.orientation > 180 || skeleton_head.orientation < -1) {
            paused = true;
        }

        if (paused && skeleton_head.orientation < 180 && skeleton_activity === "pointing") {
            paused = false;
        }

        // update ship position
        if (!paused) {
            ship_x = skeleton_head.x * 1.7;
        }
    }

    if (stage_3 && num_people == 0) {
        if (exiting) {
            var current_timestamp = new Date().getTime();
            if (current_timestamp - exiting_timestamp > 3000) {
                stage_1 = true;
                stage_2 = false;
                stage_3 = false;
                paused = false;
                exiting = false;
                exiting_timestamp = 0;
            }
        }
        else {
            exiting = true;
            exiting_timestamp = new Date().getTime();
        }
    }

    draw();
}

// Game logic
var fps = 60;
var frames_this_second = 0;
var last_fps_timestamp = 0;
var last_timestamp = 0;

var last_asteroid_timestamp = 0;
var last_bullet_timestamp = 0;

function loop(timestamp) {
    var progress = timestamp - last_timestamp;
    if (!paused) {
        update(timestamp, progress);
    }
    draw();
    if (timestamp > last_fps_timestamp + 1000) {
        fps = 0.25 * frames_this_second + (1 - 0.25) * fps;
        last_fps_timestamp = timestamp;
        frames_this_second = 0;
    }
    frames_this_second++;
    last_timestamp = timestamp;
    window.requestAnimationFrame(loop);
}

function inside(point, vs) {
    // ray-casting algorithm based on
    // http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html

    var x = point.x, y = point.y;

    var inside = false;
    for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        var xi = vs[i].x, yi = vs[i].y;
        var xj = vs[j].x, yj = vs[j].y;

        var intersect = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }

    return inside;
}

function update(timestamp, progress) {
    // collision detection between bullet and asteroids
    for (var i = bullets.length - 1; i >= 0; i--) {
        for (var j = asteroids.length - 1; j >= 0; j--) {
            if (inside(bullets[i], asteroids[j])) {
                ship_points.push({"x": bullets[i].x, "y": bullets[i].y, "time": timestamp})
                bullets.splice(i, 1);
                asteroids.splice(j, 1);
                score = score + 100;
                break;
            }
        }
    }

    // collision detection between ship and asteroids
    for (var i = asteroids.length - 1; i >= 0; i--) {
        crashed = inside({"x": ship_x, "y": ship_y}, asteroids[i]) ||
            inside({"x": ship_x + 25, "y": ship_y + 50}, asteroids[i]) ||
            inside({"x": ship_x - 25, "y": ship_y + 50}, asteroids[i])
        if (crashed) {
            ship_damages.push({"x": ship_x, "y": ship_y, "time": timestamp})
            asteroids.splice(i, 1);
            score = score - 500;
        }
    }

    // remove flying away asteroids
    for (var i = asteroids.length - 1; i >= 0; i--) {
        var outside = false;
        for (var j = 0; j < asteroids[i].length; j++) {
            if (((asteroids[i][j].x < 0 || asteroids[i][j].x > opt_in_out_width) &&
                (asteroids[i][j].y > 100)) || asteroids[i][j].y > opt_in_out_height) {
                outside = true;
                break;
            }
        }
        if (outside) {
            asteroids.splice(i, 1);
            break;
        }
    }

    // update asteroids positions
    for (var i = 0; i < asteroids.length; i++) {
        var dx = 1;
        var dy = 3;
        for (var j = 0; j < asteroids[i].length; j++) {
            asteroids[i][j].x += asteroids[i][j].x_vel;
            asteroids[i][j].y += asteroids[i][j].y_vel;
        }
    }

    // update bullets positions
    for (var i = 0; i < bullets.length; i++) {
        var new_bullet_vel = bullets[i].vel;
        var new_bullet_y = bullets[i].y - new_bullet_vel;
        bullets[i].vel = new_bullet_vel;
        bullets[i].y = new_bullet_y;
    }

    // remove ship points
    for (var i = ship_points.length - 1; i >= 0; i--) {
        if (timestamp > ship_points[i]["time"] + 1000) {
            ship_points.splice(i, 1);
        }
    }

    // remove ship damages
    for (var i = ship_damages.length - 1; i >= 0; i--) {
        if (timestamp > ship_damages[i]["time"] + 1000) {
            ship_damages.splice(i, 1);
        }
    }

    // add new asteroid per 1.5 second
    if (timestamp > last_asteroid_timestamp + 1000) {
        asteroids.push(create_asteroid());
        last_asteroid_timestamp = timestamp;
    }

    // add new bullets per second
    if (timestamp > last_bullet_timestamp + 500) {
        bullet_x = ship_x;
        bullet_y = ship_y - 10;
        bullets.push({"x": bullet_x, "y": bullet_y, "vel": 3});
        last_bullet_timestamp = timestamp;
    }
}

function create_asteroid() {
    side = 50 + 50 * Math.round(Math.random());
    N = 10 + 10 * Math.round(Math.random());
    angles = d3.range(N).map(function(i) {
        return Math.random();
    });
    sum = d3.sum(angles);
    theta_prev = 0;

    random_x = Math.floor(Math.random() * (opt_in_out_width - 0 + 1)) + 0;
    random_x_vel = -1 + 3 * Math.random();
    random_y_vel = 1 + 3 * Math.random();

    points = angles.map(function(a) {
        var rho, theta;
        theta = theta_prev + a * 2 * Math.PI / sum;
        theta_prev = theta;
        rho = d3.random.normal(side / 3, side / 32)();
        return {"x": rho * Math.cos(theta) + random_x,
                "y": rho * Math.sin(theta),
                "x_vel": random_x_vel,
                "y_vel": random_y_vel};
    });

    return points;
}

var d3line2 = d3.svg.line()
                    .x(function(d){return d.x;})
                    .y(function(d){return d.y;})
                    .interpolate("linear");

function draw() {
    d3.selectAll("text").remove();
    d3.selectAll("path").remove();
    d3.selectAll("circle").remove();
    d3.selectAll("polygon").remove();
    d3.selectAll("foreignObject").remove();

    if (stage_1) {
        opt_in_out_svg.append("svg:foreignObject")
            .attr("width", opt_in_out_width)
            .attr("height", opt_in_out_height)
            .append("xhtml:body")
            .style("margin-left", "auto")
            .style("margin-right", "auto")
            .style("background", "black")
            .style("text-align", "center")
            .style("font-family", "sans-serif")
            .style("color", "white")
            .style("font-size", "" + opt_in_out_height / 12 + "px")
            .html("<br/><br/><br/>Asteroid<br/><span><i class=\"fa fa-street-view fa-fw\"></i></span>");
    }
    else if (stage_2) {
        opt_in_out_svg.append("svg:foreignObject")
            .attr("width", opt_in_out_width)
            .attr("height", opt_in_out_height)
            .append("xhtml:body")
            .style("margin-left", "auto")
            .style("margin-right", "auto")
            .style("background", "black")
            .style("text-align", "center")
            .style("font-family", "sans-serif")
            .style("color", "white")
            .style("font-size", "" + opt_in_out_height / 12 + "px")
            .html("<br/><br/><br/>Play<br/><span><i class=\"fa fa-hand-pointer-o fa-fw\"></i></span>");
    }
    else if (stage_3) {
        // fps
        opt_in_out_svg.append("text")
            .text("FPS: " + fps.toFixed(2))
            .attr("x", opt_in_out_width-70)
            .attr("y", 20)
            .attr("font-family", "sans-serif")
            .attr("font-size", 12)
            .attr("fill", "grey");

        // ship
        opt_in_out_svg.append("path")
            .attr("d", "M " + ship_x + " " + ship_y + " l 25 50 l -50 0 z")
            .style("stroke", "steelblue")
            .style("fill", "steelblue");

        // asteroids
        for (var i = 0; i < asteroids.length; i++) {
            var points = asteroids[i];
            opt_in_out_svg.append("polygon")
                .data([points])
                .attr("points",function(d) {
                    return d.map(function(d) {
                        return [d.x, d.y].join(",");
                    }).join(" ");
                })
                .attr("fill","yellow")
                .attr("stroke-width",2);
        }

        // bullets
        opt_in_out_svg.selectAll("circle")
            .data(bullets)
            .enter()
            .append("circle")
            .attr("cx", function (d) { return d.x; })
            .attr("cy", function (d) { return d.y; })
            .attr("r", function (d) { return 10; })
            .style("fill", function(d) { return "red"; });

        if (paused) {
            opt_in_out_svg.append("svg:foreignObject")
                .attr("width", opt_in_out_width)
                .attr("height", opt_in_out_height)
                .append("xhtml:body")
                .style("margin-left", "auto")
                .style("margin-right", "auto")
                .style("background", "black")
                .style("text-align", "center")
                .style("font-family", "sans-serif")
                .style("color", "white")
                .style("font-size", "" + opt_in_out_height / 12 + "px")
                .html("<br/><br/><br/>Unpause<br/><span><i class=\"fa fa-hand-pointer-o fa-fw\"></i></span>");
        }

        // points
        for (var i = 0; i < ship_points.length; i++) {
            var pt = ship_points[i];
            opt_in_out_svg.append("text")
                .text("+100")
                .attr("x", pt["x"])
                .attr("y", pt["y"])
                .attr("font-family", "sans-serif")
                .attr("font-weight", "bold")
                .attr("font-size", 14)
                .attr("fill", "green");
        }

        // damages
        for (var i = 0; i < ship_damages.length; i++) {
            var pt = ship_damages[i];
            opt_in_out_svg.append("text")
                .text("-500")
                .attr("x", pt["x"])
                .attr("y", pt["y"])
                .attr("font-family", "sans-serif")
                .attr("font-weight", "bold")
                .attr("font-size", 14)
                .attr("fill", "red");
        }

        // score
        opt_in_out_svg.append("text")
            .text("SCORE: " + score)
            .attr("x", 40)
            .attr("y", 40)
            .attr("font-family", "sans-serif")
            .attr("font-weight", "bold")
            .attr("font-size", 30)
            .attr("fill", "green");
    }
}