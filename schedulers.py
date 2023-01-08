class ConstantScheduler():
    def __init__(self, value):
        self.value = value

    def get_value(self, epoch=None):
        return self.value

class LinearScheduler():
    def __init__(self, schedule_string):
        split_schedule = schedule_string.split(":")
        self.initial_value = float(split_schedule[0])
        self.final_value = float(split_schedule[1])
        self.ramp_init = int(split_schedule[2])
        self.ramp_end = int(split_schedule[3])

    def get_value(self, epoch):
        if epoch <= self.ramp_init:
            return self.initial_value
        elif epoch >= self.ramp_end:
            return self.final_value
        else:
            slope = (self.final_value-self.initial_value)/(self.ramp_end-self.ramp_init)
            return (epoch-self.ramp_init)*slope + self.initial_value