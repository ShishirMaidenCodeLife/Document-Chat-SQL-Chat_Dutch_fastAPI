We need to first keep the Chinook.db in the same working folder ....Hence to create Chinook.db in the same directory as this notebook:

    Save the file from this link(https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql) to your sql quring code directory by naming it as Chinook_Sqlite.sql
    Run "sqlite3 Chinook.db" in terminal
    Run ".read Chinook_Sqlite.sql"
    Test "SELECT * FROM Artist LIMIT 10;"
    
Now, Chinhook.db is in our directory.