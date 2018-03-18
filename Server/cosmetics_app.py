#!/usr/bin/env python3
'''
Cosmetics app server
'''
import time
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import re
import json
import pandas as pd
import argparse
from urllib.parse import unquote_plus
from load_data_to_mongo import display_db_stats
#from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from db_crud import DB_CRUD
from db_object import DB_Object
import json
import numpy as np

from PIL import Image
from pytesseract import image_to_string, image_to_boxes

SV_HOST_NAME = 'ec2-35-172-36-92.compute-1.amazonaws.com'
SV_PORT_NUMBER = 9000

DB_HOST_NAME = 'localhost'
DB_PORT_NUMBER = 27017

PEOPLE_DB = None
PRODUCTS_DB = None
INGREDIENTS_DB = None
COMODEGENIC_DB = None
def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


class MyHandler(BaseHTTPRequestHandler):
    store_path = os.path.join(os.curdir, 'store')
    upload_path = os.path.join(os.curdir, 'upload.jpg')
    record_data = [
        'ingredient_name',
        'ingredient_score',
        'dev_reprod_tox_score',
        'allergy_imm_tox_score',
        'cancer_score']

    raw_request = b''
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

    def do_POST(s):

        if s.path == '/barcode':
            length = s.headers['content-length']
            data = s.rfile.read(int(length))
            decoded = data.decode()
            print(decoded)
            with open(s.store_path, 'w') as fh:
                fh.write(decoded)

            s.do_HEAD()
            s.wfile.write("test")

        if s.path == '/searchterm':
            length = s.headers['content-length']
            data = s.rfile.read(int(length))
            decoded = data.decode()
            search_str = unquote_plus(str(decoded).replace('search_term=', ''))

            print('SEARCH_STR: ', search_str)

            # import ingredient score and search and return score of the search term
            with open('ewg_ingredients.json') as jsondata:
                ingredients = json.load(jsondata)
                jsondata.close()

            ewg_ingredient = pd.DataFrame.from_dict(ingredients, orient='index')
            matched_term = process.extract(
                search_str,
                ewg_ingredient['ingredient_name'],
                limit=1)[0][0]
            record_filter = ewg_ingredient.ingredient_name == matched_term
            matched_record = ewg_ingredient[record_filter][s.record_data]
            results = matched_record.to_json(orient='index')
            print(results)

            s.do_HEAD()
            s.wfile.write(results.encode("utf-8"))

        elif s.path == '/upload':
            print("recognize /upload")

            s.rfile.flush()

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
            #print(len(body))
            with open(s.upload_path, 'wb') as fh:
                fh.write(body)
            # open photo and convert to pure black and white
            img = change_contrast(Image.open(s.upload_path),100).convert('L')
            i_result = image_to_string(img)
            s.wfile.write(i_result.encode("utf-8"))
            

 # below functions is for handling chunked response for photos
    def handle_chunked_encoding(self):

        body = b''
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
                #print(trailer_line)
                while trailer_line.strip() != b'':
                    trailer_line = self.rfile.readline()
                    # Read the chunk size.
                chunk_size = self.read_chunk_size()
        return body

    def read_chunk_size(self):
        # Read the whole line, including the \r\n.
        chunk_size_and_ext_line = self.rfile.readline()
        # Look for a chunk extension.
        chunk_size_end = chunk_size_and_ext_line.decode('utf-8').find(';')
        if chunk_size_end == -1:
            # No chunk extensions; just encounter the end of line.
            chunk_size_end = chunk_size_and_ext_line.decode('utf-8').find('\r')
        if chunk_size_end == -1:
            self.send_response(400)  # Bad request.
            return -1
        return int(chunk_size_and_ext_line[:chunk_size_end], base=16)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--server_host', help='Server hostname', default=SV_HOST_NAME)
    parser.add_argument('-p', '--server_port', help='Server port', default=SV_PORT_NUMBER)
    parser.add_argument('-m', '--db_host', help='Database hostname', default=DB_HOST_NAME)
    parser.add_argument('-n', '--db_port', help='Database port', default=DB_PORT_NUMBER)
    args = parser.parse_args()

    # App databases
    PEOPLE_DB = DB_CRUD(args.db_host, args.db_port, db='capstone', col='people')
    PRODUCTS_DB = DB_CRUD(args.db_host, args.db_port, db='capstone', col='products')
    INGREDIENTS_DB = DB_CRUD(args.db_host, args.db_port, db='capstone', col='ingredients')
    COMODEGENIC_DB = DB_CRUD(args.db_host, args.db_port, db='capstone', col='comodegenic')
    display_db_stats(args.db_host, args.db_port)

    # Startup App server
    server_class = HTTPServer
    httpd = server_class((args.server_host, args.server_port), MyHandler)
    print(time.asctime(), "Server Starts - %s:%s" % (args.server_host, args.server_port))

    try:
        httpd.serve_forever()  # Server running here
    except KeyboardInterrupt:  # Server stops on keyboard interrupt
        pass

    httpd.server_close()
    print(time.asctime(), "Server Stops - %s:%s" % (args.server_host, args.server_port))
