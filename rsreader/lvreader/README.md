# *L*ow-le*V*el Reader (lvreader)

In this module, we provide some basic readers that can read images
directly from the files in the disks. 

In this section, we will introduce the basic usage of the readers.
It can be seen as a simple tutorial.
More details can be seen from the help instruction.

### PILReader
PILReader reads images based the package PIL.

- import the used modules
```angular2html
import numpy as np
from rsreader.lvreader.PILReader import PILReader
```

- define the image to be read
```angular2html
imgfile = '/directory/to/an/image.JPG'
``` 

- create a reader instance
```angular2html
reader = PILReader(imgfile)
```

- print the basic information: the number of channels and the size
```angular2html
print(reader.getNChannel(), reader.getSize())
```

- crop a patch from the original image
```angular2html
patch = reader.readPatch(startx=-10,starty=100,width=200,height=250,bandlst=[],dtype=np.uint8)
```

- read the whole image into memory
```angular2html
img = reader.readImg(bandlst=[],dtype=np.uint8)
```
### OpenCVReader
The OpenCVReader reads images based on the package OpenCV.

- import the used modules
```angular2html
import numpy as np
from rsreader.lvreader.OpenCVReader import OpenCVReader
```

- create a reader instance
```angular2html
reader = OpenCVReader(imgfile)
```

The following steps are exactly the same as the previous PILReader.

### GdalReader
The GdalReader reads images based on the package GDAL.
It is designed for geoscience and remote sensing users to read Geo-Tiff and such images,
which usually cannot be done with OpenCV or PIL.

- import the used modules
```angular2html
import numpy as np
from rsreader.lvreader.GdalReader import GdalReader
```

- create a reader instance
```angular2html
reader = OpenCVReader(imgfile)
```

The following steps are exactly the same as the previous PILReader.

### BaseReader
The BaseReader can be seen as an interface that is usually called
by other methods, though the other three basic reader can be used
independently.

- import the used modules
```angular2html
import numpy as np
from rsreader.lvreader.BaseReader import BaseReader
```

- create a reader instance
```angular2html
reader = BaseReader(imgfile)
```
or
```angular2html
reader = BaseReader(imgfile, readertype='pil')
```
or
```angular2html
from rsreader.lvreader.PILReader import PILReader
reader = BaseReader(imgfile, readertype=PILReader)
```

The following steps are exactly the same as the previous PILReader.

#### Extension
The introduce of BaseReader not only provides a unified interface
called by the other users, but also provides extension to other
low-level readers. The users can easily extend to other format
of files by implementing the interfaces in BaseReader
and creating a BaseReader instance like this.
```angular2html
from user.defined.low-level.reader import xreader
reader = BaseReader(imgfile, readertype=xreader)
```

---------------
## Test
The test instance can be seen from __test/lvreader.ipynb__
