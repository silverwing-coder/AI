''' This file reads students data from a *.json file and setup a database on sqlite 3
    Author: Sangmork park, Virginia Military Institute
    Version: Mar. 2024
    Source file: students_data.json
    Database file: students.db
'''

import sqlite3
import json

def create_students_table(cursor, data):
    keys = list(data.keys())
    values = list(data.values())
    merged = zip(keys, values)

    for element in merged:
        key1 = "id";
        key2 = list(element[1].keys())[0]
        key3 = list(element[1].keys())[1]
        key4 = list(element[1].keys())[2]
        key5 = list(element[1].keys())[3]
        key6 = list(element[1].keys())[4]
        key7 = list(element[1].keys())[5]
        key8 = list(element[1].keys())[6]
    # print(key1, key2, key3, key4,key5,key6, key7, key8)

    cursor.execute("CREATE TABLE Students ({} text, {} text, {} text, {} int, {} int, {} text, {} int, {} text"
                        .format(key1, key2, key3, key4, key5, key6, key7, key8) +")")

def initialize_students_table(cursor, data):
    keys = data.keys()
    values = data.values()
    merged = zip(keys, values)
    for element in merged:
        val1 = element[0]
        val2 = list(element[1].values())[0]
        val3 = list(element[1].values())[1]
        val4 = list(element[1].values())[2]
        val5 = list(element[1].values())[3]
        val6 = list(element[1].values())[4]
        val7 = list(element[1].values())[5]
        val8 = list(element[1].values())[6]
        # print(val1, val2, val3, val4, val5, val6, val7, val8)

        cursor.execute("INSERT INTO Students VALUES ('{}', '{}', '{}', {}, {}, '{}', {}, '{}'"
                        .format(val1, val2, val3, val4, val5, val6, val7, val8) + ")")
        cursor.execute("COMMIT")

def setupDatabaseMain():
    # connector = sqlite3.connect(':memory:')
    # connector = sqlite3.connect('students.db')
    connector = sqlite3.connect('CIS112Students.db')
    cursor = connector.cursor()

    ''' database source file : students_data.json '''
    # file = open("Resources/students_data.json")
    file = open("Resources/CIS112StudentsData.json")
    data = json.load(file)
    file.close()
    # print(list(data))
    # print(list(data.keys()))
    # print(list(data.values()))

    ''' read students data and setuup students database '''
    create_students_table(cursor, data)
    initialize_students_table(cursor, data)

    ''' check database setup '''
    cursor.execute("SELECT * FROM Students WHERE section='3'")
    print(cursor.fetchall())

    connector.close()

if __name__ == '__main__':
    setupDatabaseMain()