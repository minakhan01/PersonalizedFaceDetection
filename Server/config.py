class Config(object):
    MYSQL_DATABASE_USER = 'sql9214426'
    MYSQL_DATABASE_PASSWORD = 'EApraVLeK5'
    MYSQL_DATABASE_DB = 'sql9214426'
    MYSQL_DATABASE_HOST = 'sql9.freemysqlhosting.net'

# ndass1.pythonanywhere.com
class ProductionConfig(Config):
    ENV = "Prod"

    MYSQL_DATABASE_USER = 'ndass1'
    MYSQL_DATABASE_PASSWORD = 'FaceDetectiondb123'
    MYSQL_DATABASE_DB = 'ndass1$default'
    MYSQL_DATABASE_HOST = 'ndass1.mysql.pythonanywhere-services.com'

# 127.0.0.1:5000
class DevelopmentConfig(Config):
    ENV = "Dev"