#!/usr/bin/env python3
'''
Cosmetics app server
'''
import time
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote_plus
import cgi
import re
import json
import pandas as pd
import argparse
from load_data_to_mongo import display_db_stats
#from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from db_crud import DB_CRUD
from db_object import DB_Object, JSONEncoder
import json
import numpy as np

from PIL import Image
import PIL.ImageOps
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


def light_background(img):

    img_b = img.convert('1')

    pixels = img_b.getdata()          # get the pixels as a flattened sequence
    black_thresh = 50
    nblack = 0
    for pixel in pixels:
        if pixel < black_thresh:
            nblack += 1
    n = len(pixels)

    if (nblack / float(n)) > 0.5:

        if img.mode == 'RGBA':
            r,g,b,a = img.split()
            rgb_image = Image.merge('RGB', (r,g,b))

            inverted_image = PIL.ImageOps.invert(rgb_image)

            r2,g2,b2 = inverted_image.split()

            final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

            return final_transparent_image.invert(img).convert('L')

        else:
            return PIL.ImageOps.invert(img).convert('L')

    else:
        return img.convert(img).convert('L')


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

    def check_authentication(s, auth_str):
        """ Check geven credentials against DB"""
        in_auth = auth_str.strip().strip('Basic ')
        query = PEOPLE_DB.read({'auth': in_auth}, limit=1)
        if query.count() == 1:
            s.person_data = DB_Object.build_from_dict(query[0])
            return True
        else:
            return False

    def check_username_availability(s, search_str):
        """ Check DB to see if username is avaialble"""
        if not search_str:
            return False

        query = PEOPLE_DB.read({'user_name': search_str}, limit=1)
        if query.count() == 0:
            return True
        else:
            return False

    def create_new_user(s, recv_data):

        return PEOPLE_DB.create(DB_Object.build_from_dict(recv_data))

    def get_prod_suggestions(s, search_str):
        """ Check DB to see if username is avaialble"""
        if not search_str:
            return False

        query = PRODUCTS_DB.read(
            {'$text': {'$search': unquote_plus(search_str)}},
            limit=10,
            projection={'product_name': True, 'score': {'$meta': 'textScore'}})

        sorted_query = query.sort([('score', {'$meta': 'textScore'})])

        return [{"product_id": str(item['_id']), "product_name": item['product_name']} for item in sorted_query]

    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/json; charset=utf-8")
        s.end_headers()

    def do_AUTHHEAD(s):
        print("[do_AUTHHEAD] Send Auth Request")
        s.send_response(401)
        s.send_header('WWW-Authenticate', 'Basic realm=\"Test\"')
        s.send_header('Content-type', 'text/html')
        s.end_headers()

    def do_GET(s):

        ''' Present frontpage with user authentication. '''
        if s.headers.get('Authorization') is None:
            s.do_AUTHHEAD()

            response = {
                'success': False,
                'error': 'No auth header received'
            }

            s.wfile.write(bytes(json.dumps(response), 'utf-8'))
        elif s.check_authentication(s.headers.get('Authorization', False)):
            """Respond to a GET request."""
            s.send_response(200)
            s.send_header("Content-type", "text/json; charset=utf-8")
            s.end_headers()

            # Send user data to app
            response = s.person_data
            print('User Athenticated', json.dumps(response, cls=JSONEncoder))
            s.wfile.write(bytes(json.dumps(response, cls=JSONEncoder), 'utf-8'))

        else:
            s.do_AUTHHEAD()

            response = {
                'success': False,
                'error': 'Invalid credentials'
            }

            s.wfile.write(bytes(json.dumps(response), 'utf-8'))

    def do_POST(s):
        ''' Present frontpage with user authentication. '''
        # Check if username is available
        if s.path == '/checkuser':
            data = s.rfile.read(int(s.headers['content-length']))
            search_str = data.decode().replace('username=', '')
            print("input username:", search_str)

            response = {
                'uname_unique': s.check_username_availability(search_str),
            }

            s.do_HEAD()
            s.wfile.write(bytes(json.dumps(response), 'utf-8'))

        elif s.path == '/suggestproducts':
            data = s.rfile.read(int(s.headers['content-length']))
            search_str = data.decode().replace('search_term=', '')
            print("input search:", search_str)

            response = {
                'prod_suggestions': s.get_prod_suggestions(search_str),
            }

            s.do_HEAD()
            s.wfile.write(bytes(json.dumps(response), 'utf-8'))

        elif s.path == '/createuser':
            form = cgi.FieldStorage(
                fp=s.rfile,
                headers=s.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            recv_data = {
                'user_name': form.getvalue("user_name"),
                'name': form.getvalue("name"),
                'auth': form.getvalue("auth"),
                'age': int(form.getvalue("age")),
                'race': form.getvalue("race"),
                'skin': form.getvalue("skin"),
                'birth_sex': form.getvalue("birth_sex"),
                'has_acne': bool(form.getvalue("has_acne")),
                'acne_products': json.loads(form.getvalue("acne_products"))}
            print(recv_data)
            status = s.create_new_user(recv_data)
            if status.acknowledged:
                response = {"create_user": str(status.inserted_id)}
            else:
                response = {"create_user": None}

            s.do_HEAD()
            s.wfile.write(bytes(json.dumps(response), 'utf-8'))

        elif s.headers.get('Authorization') is None:
            s.do_AUTHHEAD()

            response = {
                'success': False,
                'error': 'No auth header received'
            }

            s.wfile.write(bytes(json.dumps(response), 'utf-8'))
            pass
        elif s.check_authentication(s.headers.get('Authorization', False)):
            # Authenticated
            if s.path == '/barcode':
                length = s.headers['content-length']
                data = s.rfile.read(int(length))
                decoded = data.decode()
                print(decoded)
                with open(s.store_path, 'w') as fh:
                    fh.write(decoded)

                response = {
                    'barcode_response': 'TEST',
                }

                s.do_HEAD()
                s.wfile.write(bytes(json.dumps(response), 'utf-8'))

            if s.path == '/searchterm':
                length = s.headers['content-length']
                data = s.rfile.read(int(length))
                decoded = data.decode()
                search_str = unquote_plus(str(decoded).replace('search_term=', ''))

                print('SEARCH_STR: ', search_str)

                # REPLACE WITH DB CALL
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
                img = change_contrast(Image.open(s.upload_path),100)
                img_final = light_background(img)
                i_result = image_to_string(img_final).split("\n")
                print(i_result)
                # find the ingredient list part from result and extract it as a list of ingredients
                start_index = 0
                end_index = 0
                for i in range (0,len(i_result)):
                    if ("Ingredient" in i_result[i]) or ("INGREDIENT" in i_result[i]) or ("Ingrédients" in i_result[i]):
                        start_index = i
                        if i_result[i+1]=="":
                            for j in range ((start_index+2),len(i_result)):
                                if i_result[j]=="":
                                    end_index = j
                                    break
                                else:
                                    end_index = len(i_result)
                        else:
                            for j in range ((start_index+1),len(i_result)):
                                if i_result[j]=="":
                                    end_index = j
                                    break
                                else:
                                    end_index = len(i_result)

                r = i_result[start_index:end_index+1]
                ingredient_list = ' '.join(r[1:]).split(',')
                print(ingredient_list)
                s.wfile.write(ingredient_list)

            pass
        else:
            s.do_AUTHHEAD()

            response = {
                'success': False,
                'error': 'Invalid credentials'
            }

            s.wfile.write(bytes(json.dumps(response), 'utf-8'))
            pass

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
