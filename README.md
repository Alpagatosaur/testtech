# Test

Create a vehicle detection application and associated French license plate recognition, based on a video provided.
Implement a complete pipeline that takes the supplied video as input, applies vehicle and plate detection, recognizes characters, and generates the same annotated video as output, with for each detected vehicle:

- a unique identifier (track_id)
- the recognized license plate number
- the detection confidence coefficient

```mermaid
flowchart TD
    id1[(Video)]
    id2([Image])
    id3{Model YOLOv5s}
    id4[[Process track ID]]
    id5[[Process read license plate]]
    id6{{write labels}}
    id7[save video]
    id1 --> id2
    id2 --> id3
    id3 --class : car--> id4
    id3 --class : license plate--> id5
    id4 --> id6
    id5 --> id6
    id6 -..-> id2
    id6 --end--> id7
```
