import optitrack.csv_reader as csv_reader

take = csv_reader.Take()

take.readCSV("../body_data/second_attempt/groundTruth.csv")

for label, body in take.rigid_bodies.items():
    print("Rigid Body:", label)
    print("ID:", body.ID)

    print("Positions:")
    for frame, position in enumerate(body.positions):
        if position is not None:
            print("Frame", frame, ":", position)

    print("Rotations:")
    for frame, rotation in enumerate(body.rotations):
        if rotation is not None:
            print("Frame", frame, ":", rotation)

    print("--------------------")
