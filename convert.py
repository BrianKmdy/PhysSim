print('running')

def check_for_data(bytes, len, location):
    for o in range(0, 4):
        for i in range(o, o + 12, 4):
            always_zero = True
            for index in range(i, len - 4, 12):
                if bytes[index] + bytes[index + 1] + bytes[index + 2] + bytes[index + 3] != 0:
                    always_zero = False

            if always_zero:
                always_always_zero = True
                for index in range(0, len - 4, 4):
                    if bytes[index] + bytes[index + 1] + bytes[index + 2] + bytes[index + 3] != 0:
                        always_always_zero = False

                if not always_always_zero:
                    print(location)
                    print(bytes)


location = 0
with open('PhysSim.exe_190808_184215.dmp', 'rb') as f:
    while True:
        r = f.read(360)
        if r:
            check_for_data(r, 360, location)
            location += 360
        else:
            break
