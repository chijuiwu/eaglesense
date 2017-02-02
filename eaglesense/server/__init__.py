"""

EagleSense web server

===
EagleSense: Tracking People and Devices in Interactive Spaces using Real-Time Top-View Depth-Sensing
Copyright © 2016 Chi-Jui Wu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import tornado.httpserver
import tornado.ioloop
import tornado.log
import tornado.web

import logging
import signal

from eaglesense.server.app import EagleSense

tornado.log.enable_pretty_logging()

is_server_down = False


def run(host, port):
    eaglesense_http_server = tornado.httpserver.HTTPServer(EagleSense())
    eaglesense_http_server.bind(address=host, port=port)
    eaglesense_http_server.start()
    logging.info("Running server @ {addr}:{port}".format(addr=host, port=port))

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    tornado.ioloop.PeriodicCallback(try_exit, 100).start()
    tornado.ioloop.IOLoop.current().start()


def sig_handler(sig, frame):
    logging.warning("Caught signal: {sig}".format(sig=sig))
    logging.warning("Exiting...")
    global is_server_down
    is_server_down = True


def try_exit():
    if is_server_down:
        tornado.ioloop.IOLoop.current().stop()
        logging.info("Exit success!!")
