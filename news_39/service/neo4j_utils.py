# encoding: utf-8
from neo4j.v1 import GraphDatabase

uri = "bolt://192.168.0.106:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "1qaz2wsx"))


def _get_department(tx, keyword, limit=2):
    query = """
    MATCH (n:Keyword {keyword: {keyword}})-[r:DepartmentLabel]-(x:Department) RETURN n,x, toInt(r.score) as score order by score desc  LIMIT %s 
    """ % limit
    rst = []
    for record in tx.run(query, keyword=keyword):
        rst.append((record["x"]["department"], record["score"]))

    return rst


def get_department(keyword, limit=2):
    with driver.session() as session:
        return session.read_transaction(_get_department, keyword, limit)


if __name__ == '__main__':
    print(get_department("肾结石", 2))

