# encoding: utf-8

from service.neo4j_utils import get_department
from service.ner_model_utils import get_ner_content
import tornado.ioloop
import tornado.web
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import json

import tornado.ioloop
import tornado.web
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
import tornado.httpserver
from functools import reduce
import pandas as pd


class DepartmentHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(20)

    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        data = json.loads(self.request.body)
        content = data.get("content", [])
        content_keywords = get_ner_content(content) if content else []
        department_stat = [reduce(lambda x, y: x+y, [get_department(_) for _ in keywords], []) for keywords in content_keywords]
        rst = []
        for content_department in department_stat:
            df = pd.DataFrame(content_department)
            max_score_department = df.groupby(0).sum().idxmax().item()
            max_score = df.groupby(0).sum().max().item()
            rst.append({"department": max_score_department, "score": max_score})

        self.write(json.dumps({"data": rst}))


def handlers():
    return tornado.web.Application([
        (r"/department", DepartmentHandler),
    ])


if __name__ == "__main__":
    app = handlers()
    print("start")
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(8080)
    http_server.start(10)
    tornado.ioloop.IOLoop.current().start()


# class DepartmentHandler(tornado.web.RequestHandler):
#     def get(self):
#         content = self.get_argument("content", "")
#         keywords = get_ner_content(content)
#         rst = [get_department(_) for _ in keywords]
#         self.write(json.dumps(rst))
#
#
# def handlers():
#     return tornado.web.Application([
#         (r"/", DepartmentHandler),
#     ])
#
#
# if __name__ == "__main__":
#     app = handlers()
#     app.listen(8888)
#     tornado.ioloop.IOLoop.current().start()
#
#     # sockets = tornado.netutil.bind_sockets(8888)
#     # tornado.process.fork_processes(0)
#     # server = HTTPServer(app)
#     # server.add_sockets(sockets)
#     # IOLoop.current().start()
#     print("start")
