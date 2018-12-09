import numpy as np
from cloudant import couchdb
import json



#A function to test db connection.
def openCouchSession():
    with couchdb('admin', 'isthisroomoccupied', url='http://127.0.0.1:5984/') as client:
    # Context handles connect() and disconnect() for you.
    # Perform library operations within this context.  Such as:
        print(client.all_dbs()) 

#Gets all documents from the given database
def fetchAll(dbname):
    with couchdb('admin', 'isthisroomoccupied', url='http://127.0.0.1:5984/') as client:
        my_database = client[dbname]
        docs = json.loads('{"docs": []}')
        for document in my_database:
            docs['docs'].append(document)
        return docs

#gets the values from the corresponding values for the key 'Occupancy' of the given database
def fetchLabels(dbname):
    with couchdb('admin', 'isthisroomoccupied', url='http://127.0.0.1:5984/') as client:
        db = client[dbname]
        docs = []
        for document in db:
            docs.append(document['Occupancy'])
        return docs

#Gets the 'values' from the given database
def fetchTimes(dbname):
    with couchdb('admin', 'isthisroomoccupied', url='http://127.0.0.1:5984/') as client:
        db = client[dbname]
        docs = []
        for document in db:
            docs.append(document['date'])
        return docs

