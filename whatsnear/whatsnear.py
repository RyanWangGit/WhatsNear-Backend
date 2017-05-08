from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import json


class WhatsNearHTTPHandler(BaseHTTPRequestHandler):
    def parse_parameters(self):
        path = ''
        params = {}
        if '?' in self.path:
            path, args = self.path.split('?')
            for param in args.split('&'):
                name, value = param.split('=')
                params[name] = value

        return path, params

    def do_GET(self):
        path, params = self.parse_parameters()

        # prepare the header
        self.protocol_version = 'HTTP/1.1'
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # prepare the content
        if path == '/query':
            self.wfile.write(self.respond_query(params))

    def respond_query(self, params):
        points = json.loads(params['points'])
        return 'Hello World!'



def start_server(port=8080):
    http_server = HTTPServer(('127.0.0.1', int(port)), WhatsNearHTTPHandler)
    http_server.serve_forever()
