import optitrack.csv_reader as csv_reader
import json
import sys

def main():

    if len(sys.argv) > 2:
        file_to_read = sys.argv[1]
        label = sys.argv[2]
    else:
        print("Not enough arguments")
        exit(1)

    take = csv_reader.Take()
    take.readCSV("../body_data/"+file_to_read+".csv")

    # Specify the output file path
    output_file = "../body_data/"+file_to_read+"1.json"

    # Create a list to store the data for each frame
    frames_data = []

    # Iterate over each frame
    for frame in range(len(next(iter(take.rigid_bodies.values())).positions)):
        frame_data = {"Frame": frame, "keypoints": []}

        keypoints = []
        # Iterate over each rigid body
        for rigid_body_label, rigid_body in take.rigid_bodies.items():
            positions = rigid_body.positions

            # Check if the rigid body has a position for the current frame
            if positions[frame] is not None:
                # Write the rigid body label and its position
                struct = {"Label": rigid_body_label, "Position": positions[frame]}
                if label==True:
                    keypoints.append(struct)
                else:
                    keypoints.append(positions[frame])

        frame_data["keypoints"] = keypoints
        frames_data.append(frame_data)

    print(f"Positions saved to {output_file}")

    # Write the frames data to the output file as a JSON array
    with open(output_file, "w") as file:
        json.dump(frames_data, file, indent=4)

if __name__ == '__main__':
    main()
