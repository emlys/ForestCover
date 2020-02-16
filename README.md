# ForestCover
CSE 140 forest cover type prediction project

## Data Scaling
The original dataset has columns with values ranging a few orders of magnitude. To scale them all to the same range, e.g. [0, 1] or [-1, 1], it's necessary to know what they represent and what the minimum and maximum values are. Here I describe the data and my decisions about scaling.
### Elevation
According to the US Forest Service, the elevation in the [Rawah](https://www.fs.usda.gov/detail/arp/specialplaces/?cid=fsm91_058256), [Neota](https://www.fs.usda.gov/recarea/arp/recarea/?recid=80806), [Comanche Peak](https://www.fs.usda.gov/detail/arp/specialplaces/?cid=fsm91_058246), and [Cache la Poudre](https://www.fs.usda.gov/recarea/arp/recarea/?recid=80801) wilderness areas ranges from 6,100 to 13,000 feet (1,859 to 3,962 meters). To account for likely imprecision of these values, I am rounding the lower bound down and the upper bound up, and assume that all elevation values in the dataset will be in the range of 1800 to 4000 meters.
### Aspect
Aspect describes the compass direction that the slope faces and so ranges from 0 to 360. The full range of values is present in the dataset.
### Slope
Slope can take values in the range 0 to 90 (completely flat to completely vertical). The maximum value in the datasets is 66.
### Horizontal distances
These could in theory be any positive number but the bounds of the study area limit them to reasonable values. The ranges seen in the dataset are:
Horizontal distance to hydrology: [0, 1,397]; I will assume 1,400 is the max.
Horizontal distance to roadways: [0, 7117]; I will assume 7200 is the max.
Horizontal distance to fire points: [0, 7173]; I will assume 7200 is the max.
### Vertical distance to hydrology
This is the only variable which has negative values. The range seen in the dataset is [-173, 601]. It seems likely that values outside this range could occur in another test set, but for now I'll assume that this is the complete range.
### Hillshade at 9am, noon, and 3pm
Hillshade is a measure of the amount of sunlight falling on a surface at a given time. It is represented as a grayscale value, a number in the range [0, 255].
### Wilderness areas
The data does not evenly represent each wilderness area: in the training set, there are 3597 samples from area 1 (Rawah), 499 from area 2 (Neota), 6349 from area 3 (Comanche Peak), and 4675 from area 4 (Cache la Poudre). So, we could try to account for this. But it doesn't seem likely that the wilderness area is a very important factor to the prediction.

