function EagleSenseWebSocket() {
    this.skeleton_ws_url = "ws://" + window.location.host + "/ws/skeleton/";
    this.skeleton_ws = new WebSocket(this.skeleton_ws_url);
}

EagleSenseREST.prototype.connect = function(
    onopen_callback, onmessage_callback, onclose_callback, onerror_callback) {

    this.skeleton_ws.onopen = function(event) {
        console.log("Connected to the EagleSense skeleton stream");
        onopen_callback();
    };

    this.skeleton_ws.onmessage = function(event) {
        try {
            data = JSON.parse(event.data);
            onmessage_callback(data);
        } catch(e) {
            return;
        }
    };

    this.skeleton_ws.onclose = function(event) {
        onclose_callback();
    };

    this.skeleton_ws.onerror = function(event) {
        onerror_callback();
    };
};
