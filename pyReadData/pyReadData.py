# -*- coding: utf-8 -*-

def ReadOracle1(tablename, user, password, ip, dbname):
    ## based on cx_Oracle dependence oracle client instance
    
    import timeit
    import cx_Oracle 
    import pandas as pd

    link = user + '/' + password + '@' + ip + '/' + dbname
    comm = 'select * from ' + tablename
    
    conn = cx_Oracle.connect(link) 
    cur = conn.cursor() 
    cur.execute(comm)
    x0 = cur.fetchall()
    cols0 = [des[0] for des in cur.description]
    tab0 = pd.DataFrame(x0, columns=cols0)
    cur.close()
    #conn.commit()
    conn.close()
    
    #tab0.to_csv('***.csv', sep=',', index=False, header=True)
    return(tab0)

def readSQLdb(tablename, user, password, host, port, dbname, charset = 'utf8'):
    
    comm = 'select * from ' + tablename
    
    import MySQLdb
    conn= MySQLdb.connect(
                          host = host,
                          port = port,
                          user = user,
                          passwd = password,
                          db = dbname,
                          charset = charset
                          )
    cur = conn.cursor()
    #cur.execute("SET NAMES utf8")
    cur.execute(comm)
    info = cur.fetchall()
    cur.close()
    ##conn.commit()
    conn.close()
    return(info)

if __name__ == '__main__':
    print 'fix the mysqldb info return'
    
    