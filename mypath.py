
# cslab cluster
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'prostate':
            return '/data1/zyy/code/datasets/prostate'

        elif database == 'refuge':
            return '/data1/zyy/code/datasets/refuge'

        elif database == 'dgs':
            return '/data1/zyy/code/datasets/dgs1'

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'models/'

