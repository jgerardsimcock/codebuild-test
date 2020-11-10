import pyodbc


class SQL_connector:
    """
    A connector for interfacing with SQL data directly in Python, exectued using PYODBC.


    Class Vars:
        driver: (str) the SQL driver to use
        server: (str) the address of the SQL server
        database : (str) the name of the SQL database
        user: (str) the user name with which to access the data
        password: (str) the password for accessing the data
        trusted: (bool) whether or not to use a trusted connection

    """

    def __init__(self, conn_dict):
        for attr in ["driver", "server", "database", "user", "password"]:
            if attr not in conn_dict:
                raise ValueError(f"{attr} not found")
            else:
                setattr(self, attr, conn_dict[attr])

        if "trusted" not in conn_dict:
            self.trusted = False
        else:
            self.trusted = conn_dict["trusted"]

        self.generate_connector()

    def generate_connector(self):
        """
        Compile the PYODBC connector using class variables.

        Args:
            None
        Returns:
            None
        """
        conn_string = ""

        for attr in ["driver", "server", "database", "user", "password", "trusted"]:
            new_clause = f"{attr}={getattr(self, attr)};"
            conn_string += new_clause

        self.connector = pyodbc.connect(conn_string)
