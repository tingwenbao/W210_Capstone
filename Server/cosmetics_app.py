import time
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import re
import json
import pandas as pd
import argparse
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


HOST_NAME = 'ec2-35-172-36-92.compute-1.amazonaws.com'
PORT_NUMBER = 9000


class BufferedReadFile(object):

    def __init__(self, real_file):
        self.file = real_file
        self.buffer = ""

    def read(self, size=-1):
        buf = self.file.read(size)
        self.buffer += buf
        return buf

    def readline(self, size=-1):
        buf = self.file.readline(size)
        self.buffer += buf
        return buf

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class MyHandler(BaseHTTPRequestHandler):
    store_path = os.path.join(os.curdir, 'store')
    upload_path = os.path.join(os.curdir, 'upload.jpg')
    record_data = [
        'ingredient_name',
        'ingredient_score',
        'dev_reprod_tox_score',
        'allergy_imm_tox_score',
        'cancer_score']

    raw_request = ''
    response_code = 500
    response_body = ''

    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/json; charset=utf-8")
        s.end_headers()

    def do_GET(s):
        """Respond to a GET request."""
        s.send_response(200)
        s.send_header("Content-type", "text/json; charset=utf-8")
        s.end_headers()
        s.wfile.write("<html><head><title>Title goes here.</title></head>")
        s.wfile.write("<body><p>This is a test.</p>")
        # If someone went to "http://something.somewhere.net/foo/bar/",
        # then s.path equals "/foo/bar/".
        s.wfile.write("<p>You accessed path: %s</p>" % s.path)
        s.wfile.write("</body></html>")

    def handle_one_request(self):
        # Wrap the rfile in the buffering file object so that the raw header block
        # can be written to stdout after it is parsed.
        self.rfile = BufferedReadFile(self.rfile)
        BaseHTTPServer.BaseHTTPRequestHandler.handle_one_request(self)

    def do_POST(s):

        if s.path == '/barcode':
            length = s.headers['content-length']
            data = s.rfile.read(int(length))
            decoded = data.decode()
            print (decoded)
            with open(s.store_path, 'w') as fh:
                fh.write(decoded)

            s.do_HEAD()
            s.wfile.write("test")

        if s.path == '/searchterm':
            length = s.headers['content-length']
            data = s.rfile.read(int(length))
            decoded = data.decode()
            print (decoded)
            search_str = re.search('(?<==)\w+', str(decoded))
            print (search_str.group(0))
            # import ingredient score and search and return score of the search term
            with open('ewg_ingredients.json') as jsondata:
                ingredients = json.load(jsondata)
                jsondata.close()

            ewg_ingredient = pd.DataFrame.from_dict(ingredients, orient='index')
            matched_term = process.extract(
                search_str.group(0),
                ewg_ingredient['ingredient_name'],
                limit=1)[0][0]
            record_filter = ewg_ingredient.ingredient_name == matched_term
            matched_record = ewg_ingredient[record_filter][s.record_data]
            results = matched_record.to_json(orient='index')
            print (results)

            s.do_HEAD()
            s.wfile.write(results.encode("utf-8"))

        elif s.path == '/upload':
            print("recognize /upload")

            MyHandler.raw_request = s.rfile.buffer
            s.rfile.buffer = ''

            if s.headers.get('Transfer-Encoding', '').lower() == 'chunked':
                if 'Content-Length' in s.headers:
                    raise AssertionError
                body = s.handle_chunked_encoding()
            else:
                length = int(s.headers.get('Content-Length', -1))
                body = s.rfile.read(length)

            if s.headers.get('Content-Encoding', '').lower() == 'gzip':
                # 15 is the value of |wbits|, which should be at the maximum possible
                # value to ensure that any gzip stream can be decoded. The offset of 16
                # specifies that the stream to decompress will be formatted with a gzip
                # wrapper.
                print("gzip")
                body = zlib.decompress(body, 16 + 15)

            MyHandler.raw_request += body

            s.send_response(200)
            s.end_headers()
            print (len(body))
            with open(s.upload_path, 'wb') as fh:
                fh.write(body)

 # below functions is for handling chunked response for photos
    def handle_chunked_encoding(self):

        body = ''
        chunk_size = self.read_chunk_size()
        while chunk_size > 0:
            # Read the body.
            data = self.rfile.read(chunk_size)
            chunk_size -= len(data)
            body += data
            # Finished reading this chunk.
            if chunk_size == 0:
                # Read through any trailer fields.
                trailer_line = self.rfile.readline()
                print (trailer_line)
                while trailer_line.strip() != '':
                    trailer_line = self.rfile.readline()
                    # Read the chunk size.
                chunk_size = self.read_chunk_size()
        return body

    def read_chunk_size(self):
        # Read the whole line, including the \r\n.
        chunk_size_and_ext_line = self.rfile.readline()
        # Look for a chunk extension.
        chunk_size_end = chunk_size_and_ext_line.find(';')
        if chunk_size_end == -1:
            # No chunk extensions; just encounter the end of line.
            chunk_size_end = chunk_size_and_ext_line.find('\r')
        if chunk_size_end == -1:
            self.send_response(400)  # Bad request.
            return -1
        return int(chunk_size_and_ext_line[:chunk_size_end], base=16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--host', help='Server hostname', default=HOST_NAME)
    parser.add_argument('-p', '--port', help='Server port', default=PORT_NUMBER)
    args = parser.parse_args()

    server_class = HTTPServer
    httpd = server_class((args.host, args.port), MyHandler)
    print (time.asctime(), "Server Starts - %s:%s" % (args.host, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print (time.asctime(), "Server Stops - %s:%s" % (args.host, args.port))
