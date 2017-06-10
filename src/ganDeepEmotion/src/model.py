from sklearn.externals import joblib

class Model(object):

    def dump(self, file):
        joblib.dump([p.get_value() for p in self.params], file)

    def load(self, file):
        params_values = joblib.load(file)
        for (p, value) in zip(self.params, params_values):
            p.set_value(value)

    #def load_old(self, gen_file, dis_file):
        #gen_params_values = joblib.load(gen_file)
        #disclass_params_values = joblib.load(dis_file)

        #for (p, value) in zip(self.gen_params, gen_params_values):
            #p.set_value(value)
        #for (p, value) in zip(self.disclass_params, disclass_params_values):
            #p.set_value(value)
