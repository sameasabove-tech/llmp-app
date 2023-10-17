from tinydb import TinyDB, Query
from pathlib import Path


class DocumentDatabase:
    """A class that loads the document database that corresponds to the endpoint given during instantiation.
    """
    def __init__(self, endpoint):
        """
        Initializes an instance of TinyDB.
        """
        self.endpoint = endpoint 
        self.__load_database__()


    def __load_database__(self) -> None: # should this return a dict?
        """Loads the document database depending on the type application instantiated.

        Args:
            endpoint (str): The application endpoint that corresponds to the database we will be reading and writing from.
        """
        pwf_path = Path(__file__)
        self.db_json = TinyDB(pwf_path.parent.parent / f'storage/{self.endpoint}_db.json')
        print(f'{self.endpoint} database has been loaded. \n')
        # print(self.db_json)


    def write_to_database(self, new_insert) -> None:
        """Append a row to this document database.

        Args:
            new_insert (dict): Row of document database to be appended.
        """
        self.db_json.insert(new_insert)

    def query_database(self): #, document_row) -> list[dict]:
        """_summary_

        Returns:
            document_row[dict]: 
        """
        # Instantiate a Query object from TinyDB
        # ie.
        # db = TinyDB("todo_db.json")
        # Todo = Query()
        # db.all() --> to see entire doc db
        # db.search(Todo.name == 'Book') -> list of items matching query
        # db.get(Todo.name == 'Book') -> returns one item matching item 
        # db.contains(Todo.name == 'Copies')
        pass






