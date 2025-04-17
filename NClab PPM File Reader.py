from typing import Callable

# Looked up some Numpy functions, like np.reshape()
def render(P: list[str]):
    
    if not (error:= _checkvalues(P))[0]:
        raise ValueError(error[1])
    
    # Read the header
    header = P[0]

    # Read width, height, and max color value
    width = int(P[1])
    height = int(P[2])
    # If header is P2 or P3, it will have custom value. Else P1, it will be 1
    max_color = int(P[3]) if header != 'P1' else 1

    try:
        # Read pixel data
        pixel_array = np.array(list(map(int, P[4:])), dtype=np.uint8)
        pixel_array = pixel_array.reshape((height, width, 3)) if header == 'P3' else pixel_array.reshape((height, width))
    except:
        raise ValueError("Non-Integer pixel value")

    Max = np.max(pixel_array)
    Min = np.min(pixel_array)
    # Check if any value is outside the range (0, max_color)
    if not (0 < Max <= max_color) or Min < 0:
        raise ValueError(f"Pixel values must be in the range 0 to {max_color}")
    if max_color != 255:
        pixel_array = (pixel_array / max_color * 255).astype(np.uint8)

    return pixel_array

def writemapfile(path:str, data: 'np.ndarray', cmax:int=255):
    """Writes Numpy array 'data' to a Map file"""
    w = t.time()
    try:
        # Write file 'data' to 'path'
        success = _writemapfunc(path, data, cmax)
    except Exception as e:
        raise Exception(f"Could not write to '{path.split('.')[-1].upper()}'.\n{e}")
    if not success[0]:
        raise Exception(f"Could not write to '{path.split('.')[-1].upper()}'.\n{success[1]}")
    fin_w = t.time()
    if called_from_main(): print("WRITE:", fin_w - w)


def _writemapfunc(path: str, data: 'np.ndarray', cmax=255):
    # {extensions} is a dictionary because exts['pbm'] = 'P1'. Easy.
    exts = {'pbm': 'P1', 'pgm': 'P2', 'ppm': 'P3'}
    
    # Access types
    global file_types
    ftype = file_types[id(data)]

    # Get file ext:
    ext = path.split('.')[-1]
    if ext not in exts:
        return ValueError('File type not supported.')
    elif ext != ftype:
        return ValueError('File type and data type do not match.')
    
    if cmax != 255:
        data = (data / cmax * 255).astype(np.uint8)
    if ftype == 'pbm':
        cmax = 1
    
    with open(path, 'w') as f:
        # Get Width and Height
        H = height(data); W = width(data)
        
        f.write(f"{exts[ext]}\n{W} {H}") # No '\n' character at the end
        # It P2 or P3, it needs the max color value
        if ftype != 'pbm': f.write(f"\n{cmax}")
        for i in range(H):
            # P3 needs another inner loop
            if ftype == 'ppm':
                f.write('\n')
                for j in range(W):
                    # Need to turn d to a string for it to be used.
                    # Using regular d returns an error, which will return False if it isn't between '0' and cmax.
                    # Will work for anything that isn't a string by default.
                    f.write(' '.join([ str(d) if 0 <= d <= cmax else d for d in data[i][j] ]) + ' ')
            else:
                # Almost same thing as above
                f.write('\n' + ' '.join([ str(d) if 0 <= d <= cmax else d for d in data[i] ]))
    return [True]

def _checkvalues(data: 'np.ndarray'):
    """Checks the data to see if it is a valid image"""
    # If data is not a list, then raise an exception:
    if not isinstance(data, list):
        return False, 'Data is not a list.'
    
    # If data[0] is not 'P2', then raise an exception:
    if data[0] not in ['P1', 'P2', 'P3']:
        return False, 'Invalid Header'
    
    if data[0] == 'P1':
        if len(data) < 3:
            return False, 'Not enough data'
    else:
        # If there are less than 4 items in the list data, then raise an exception:
        if len(data) < 4:
            return False, 'Not enough data'
    
    # If data[1] data[2] do not represent an integer, then raise an exception:
    if not data[1].isdigit() or not data[2].isdigit():
        return False, 'Width and Height must be integers'
    
    # If it is PGM or PPM, make sure data[3] is digit too
    if data[0] != 'P1' and not data[3].isdigit():
        return False, 'Max color value must be an integer'
    
    # Nothing is wrong, return True
    return True, None
    
def hflip(data: 'np.ndarray', tile_size:int=1):
    """
    Flips an image horizontally
    - Allows a tile-size block to keep together as the image is flipped
    """

    h = t.time()
    # Access types
    global file_types
    ftype = file_types.get(id(data), None)

    if type(tile_size) != int:
        raise ValueError("tile_size must be an integer")
    if tile_size < 1 or tile_size > width(data):
        raise ValueError(f"tile_size must lie between 1 and {width(data)} (width of image)")
    if not (width(data) / tile_size).is_integer():
        raise ValueError("tile_size must divide evenly into the width of the image")

    if tile_size != 1:
        # Flip tiles horizontally
        cols = width(data) // tile_size
        reshaped = data.reshape(height(data), cols, tile_size, -1)
        flipped = reshaped[:, ::-1, :, :]
        new_data = flipped.reshape(data.shape)
    else:
        # Flip the entire array horizontally
        new_data = data[:, ::-1, ...]
    
    fin_h = t.time()
    if called_from_main(): print("HORIZONTAL FLIP:", fin_h - h)
    if ftype:
        file_types[id(new_data)] = ftype

    # Return new_data:
    return new_data
    
def vflip(data: 'np.ndarray', tile_size:int=1):
    """
    Flips an image vertically
    - Allows a tile-size block to keep together as the image is flipped
    """

    v = t.time()
    # Access types
    global file_types
    ftype = file_types.get(id(data), None)

    if type(tile_size) != int:
        raise ValueError("tile_size must be an integer")
    if tile_size < 1 or tile_size > height(data):
        raise ValueError(f"tile_size must lie between 1 and {height(data)} (height of image)")
    if not (height(data) / tile_size).is_integer():
        raise ValueError("tile_size must divide evenly into the height of the image")
    
    if tile_size != 1:
        # Flip tiles horizontally
        rows = height(data) // tile_size
        reshaped = data.reshape(rows, tile_size, width(data), -1)
        flipped = reshaped[::-1, ...]
        new_data = flipped.reshape(data.shape)
    else:
        # Flip the entire array horizontally
        new_data = data[::-1, ...]

    fin_v = t.time()
    if called_from_main(): print("VERTICLE FLIP:", fin_v - v)
    if ftype:
        file_types[id(new_data)] = ftype

    # Return the result:
    return new_data


def rotate(data: 'np.ndarray', deg:str|int=90, tile_size:list[int]=[1, 1], only_rotate_tile:bool=False):
    """
    Rotates a Port Map image by a multiple of 90 degrees
    - Allows a tile-size block of (Height, Width) to rotate
    """
    # This function was very difficult to add tile functionality.
    # Especially the rotate 90 and 270, as 180 just flips in both directions.
    # AI helped me learn about the np.transpose() function, but the implementation was all me.
    
    r = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    deg = int(deg)
    if deg % 90 != 0:
        raise ValueError("Invalid angle. Must be divisible by 90 degrees.")
    
    # Been thinking about making these three code comments an actual part of the code, and removing their counterparts
    """
    This error variable will get the remainder between each side length and tile_size.
    If it's indivisible, it will set error to the respective string using `and`.
    Then using `or`, it will compute both and if either return an 'error', then it will be raised. Test it!
    """
    # error = (width(data) % tile_size[1] != 0 and "tile_size must divide evenly into the width of the image") or (height(data) % tile_size[0] != 0 and "tile_size must divide evenly into the height of the image")
    # if error:
    if height(data) % tile_size[0] != 0 or width(data) % tile_size[1] != 0:
        # raise ValueError(f"Invalid tile_size. {error}")
        raise ValueError("Invalid tile_size. Maybe swap the values' places?")
    
    if type(only_rotate_tile) != bool:
        raise ValueError("only_rotate_tile must be a bool value.")
    
    # This will make it a number 0-3
    rotate = int(deg // 90) % 4
    
    # This lambda function allows me to easily reshape the array, depending on file type
    shape = lambda *sizes: list(sizes) + [3] if ftype == 'ppm' else sizes
        
    # Get the number of rows and columns
    rows = height(data) // tile_size[0]
    cols = width(data) // tile_size[1]

    if rotate == 2:

        new_data = data.reshape(shape(rows, tile_size[0], cols, tile_size[1]))

        if only_rotate_tile:
            new_data = new_data[:, ::-1, :, ::-1, :]
        else:
            new_data = new_data[::-1, :, ::-1, :, :]

        new_data = new_data.reshape(shape(height(data), width(data)))

    elif rotate % 2 == 1:

        if rotate == 1: data = hflip(data, tile_size[1])
        
        # This breaks it up into tile so I can move them around easily
        new_data = data.reshape(shape(rows, tile_size[0], cols, tile_size[1]))

        if only_rotate_tile:
            # Swap the tiles' data, but do not reorder them(the 0 and 2 are same places)
            new_data = new_data.transpose(0, 3, 2, 1, 4)

            if rotate == 1:
                # Reverse the data in the height axis
                new_data = new_data[:, ::-1, ::-1, :, :]
            elif rotate == 3:
                # Reverse the data in the width axis
                new_data = new_data[:, :, :, ::-1, :]
            
            # Reshape it back to normal, but now it has flipped width and height
            new_data = new_data.reshape(shape(rows*tile_size[1], cols*tile_size[0]))

        else:
            new_data = new_data.transpose(2, 1, 0, 3, 4)
            new_data = new_data.reshape(shape(cols*tile_size[0], rows*tile_size[1]))
            
            if rotate == 3: new_data = hflip(new_data, tile_size[1])

    file_types[id(new_data)] = ftype
    fin_r = t.time()
    if called_from_main(): print("ROTATE:", fin_r - r)

    return new_data

def shrink(data: 'np.ndarray', scale: int|float=0.5) -> list:
    """Shrinks the image the specified amount"""
    s = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    if not type(scale) in (float, int):
        raise TypeError("scale needs to be of type 'int' or 'float'")
    
    # Making it greater than 1 makes it easier to work with
    if scale < 1:
        # scale cannot be 0
        if scale <= 0: raise ValueError('scale must be greater than 0')
        scale = 1 / scale
        if int(scale) != scale: raise ValueError('scale has to be an inverse of a whole number, or a whole number')
        scale = int(scale)
        
    shape = lambda *sizes: list(sizes) + [3] if ftype == 'ppm' else sizes

    new_data = np.zeros(shape(height(data)//scale, width(data)//scale), dtype=np.uint8)

    for i in range(0, len(data), scale):
        for j in range(0, len(data[0]), scale):
            sliced = data[i:i+scale, j:j+scale, ...]

            if sliced.shape[:2] != (2, 2):
                continue
            
            val = np.sum(sliced, axis=(0, 1)) / scale**2
            new_data[i//scale][j//scale] = val if ftype != 'pbm' else np.round(val)

    file_types[id(new_data)] = ftype
    fin_s = t.time()
    if called_from_main(): print("SCALED:", fin_s - s)

    return new_data

def tile(data: 'np.ndarray', nx: int, ny: int) -> list[int]:
    """Tiles map data nx and ny times"""

    i = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    nx, ny = int(nx), int(ny)
    if not (nx and ny):
        raise ValueError("Values must be greater than 0")
    if ftype == 'ppm':
        new_data = np.tile(data, (ny, nx, 1))
    else:
        new_data = np.tile(data, (ny, nx))

    file_types[id(new_data)] = ftype
    fin_i = t.time()
    if called_from_main(): print("TILE:", fin_i - i)

    return new_data

def tile_functions(data: 'np.ndarray', nx, ny, functions):
    """This function does NOT work. It may not even get finished, but I didn't want to delete it yet."""

    i = t.time()
    nx, ny = int(nx), int(ny)
    if not (nx and ny):
        raise ValueError("Values must be greater than 0")
    
    new_data = np.tile(data, (ny, nx, 1))

    if not functions:
        pass
    elif len(functions) != ny or len(functions[0]) != nx:
        raise ValueError("Functions must be the same size as the tile")
    else:
        if isinstance(data[0][0], np.ndarray):
            func_data = np.zeros(ny*nx*len(data)*len(data[0])*3, np.uint8).reshape((ny, nx, len(data), len(data[0]), 3))
        else:
            func_data = np.zeros(ny*nx*len(data)*len(data[0]), np.uint8).reshape((ny, nx, len(data), len(data[0])))
        temp_data = [ [ [ ] ] * nx ] * ny
        for y in range(ny):
            for x in range(nx):
                f = functions[y][x]
                # If a function is given, apply it to the tile
                if f:
                    params = inspect.signature(f).parameters
                    keys = list(params.keys())
                    given = {}
                    hints = f.__annotations__
                    if len(keys) > 1:
                        print(f"Choose values for: {f.__name__}")
                    for p in keys[1:]:
                        hint = hints[p].__args__
                        default = False
                        if params[p].default is not inspect._empty:
                            print(f"  Default value: {params[p].default}")
                            default = True
                        print(f"  Expected input: {hints[p]}")
                        if any(item in (str, int, float) for item in hint):
                            answer = input(f"  Input argument for '{p}': ")
                            if any(item in (int, float) for item in hint):
                                try:
                                    answer = int(answer)
                                except:
                                    if not default:
                                        raise ValueError(f'{', '.join([ h.__name__ for h in hint])} expected, but not given.')
                        elif any(item in (tuple, list) for item in hint):
                            # This is where I left off. Trying to get lists and tuples as actual iterables instead of strings
                            answer = input(f"  Input argument for '{p}': ")
                            answer = [ s for s in answer if s.isalpha() ]
                            print(repr(answer))
                        elif hint == dict:
                            answer = 'pass'
                        if not default and not answer:
                            raise ValueError('Must give parameter if no default value exists.')
                        given[p] = answer or params[p].default

    fin_i = t.time()
    if called_from_main(): print("TILE:", fin_i - i)

    return new_data

def subimage(data: 'np.ndarray', start:list[int], end:list[int], **functions:dict[Callable, tuple]) -> list:
    """
    Returns a sub-image with pixel ranges (ax,ay) x (bx,by)
    
    You can also give a function and its required values to instead replace
    the specified range with the data modified by the function.
    - Use kwargs: function=(args), function2=(args2), etc
        - Note: The data arg will not be accepted, only other positional and kw args.
        - If you want to use kwargs for the function, make another dictionary with
            all the kwargs you want to use as the last arg.
    """

    s = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    ax, ay = start; bx, by = end
    if any(map(lambda x: type(x) != int, (ax, ay, bx, by))):
        raise TypeError("Coordinates must be integers.")
    elif bx <= ax or by <= ay:
        raise ValueError("b coordinates must be greater than a coordinates.")
    elif ( ay < 0 or by > height(data) or ax < 0 or bx > width(data) ):
        raise ValueError("Coordinates must be inside the max length of x and y.")
    
    new_data = data[ay:by, ax:bx, ...]

    if functions:
        # Separate the function from the args
        functions = list(functions.items())
        file_types[id(new_data)] = ftype

        for f, a in functions:
            # Get the function
            fun = globals().get(f, None)

            if not fun:
                raise ValueError(f"Function {f} does not exist.")

            # Get the args
            if type(a[-1]) == dict:
                args = a[:-1]
                kwargs = a[-1]
            else:
                args = a
                kwargs = {}
                
            try:
                placeholder = fun(new_data, *args, **kwargs)
                data[ay:by, ax:bx, ...] = placeholder
            except Exception as e:
                print(e)
            del file_types[id(new_data)]
            new_data = data


    file_types[id(new_data)] = ftype
    fin_s = t.time()
    if called_from_main(): print("SUBIMAGE:", fin_s - s)

    return new_data

def makegray(data: 'np.ndarray'):
    """Turns a PPM image grayscale"""

    g = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    if ftype != 'ppm':
        raise ValueError('Only ppm files can be turned grayscale')
    
    red_weight, green_weight, blue_weight = 0.299, 0.587, 0.114
    weights = np.array([red_weight, green_weight, blue_weight], np.float32)
    colored_data = data * weights
    new_data = np.sum(colored_data, axis=2).astype(np.uint8)
    
    file_types[id(new_data)] = 'pgm'
    fin_g = t.time()
    if called_from_main(): print("MADEGRAY:", fin_g - g)

    return new_data

def makebinary(data: 'np.ndarray', threshold:int=100):
    """Turns a PGM or PPM image into a binary image"""

    b = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    if ftype == 'ppm':
        data = makegray(data)
    elif ftype == 'pbm':
        return data

    if type(threshold) != int:
        raise ValueError("Threshold must be an integer between 1 and 255")

    if threshold < 1 or threshold > 255:
        raise ValueError("Threshold must be between 1 and 255")

    new_data = data > threshold

    file_types[id(new_data)] = 'pbm'
    fin_b = t.time()
    if called_from_main(): print("MAKEBINARY:", fin_b - b)

    return new_data

def brightness(data, percent:float|str|int=100):
    """Changes the brightness percentage of the image"""

    b = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    if ftype == 'pbm':
        raise ValueError('pbm files cannot change brightness')
    
    if type(percent) == str:
        if percent.endswith('%'):
            percent = percent[-1]
        try:
            percent = float(percent)
        except:
            raise ValueError("Percent must be a number")
    p:float = percent / 100
    if p > 1:
        p = 1 - (1 / p)
    else:
        p = -1 + p

    # If brightening it, we need to invert the values, so they don't go over 255
    inverted_data = 255 - data if p > 0 else data
    # Now multiply it by the brightness factor
    adjusted_data = inverted_data * p
    # Add the values back, and convert to integers
    new_data = (adjusted_data + data).astype(np.uint8)

    file_types[id(new_data)] = ftype
    fin_b = t.time()
    if called_from_main(): print("BRIGHTNESS:", fin_b - b)

    return new_data

# This function doesn't work how I want it, but I am currently trying to fix it
def filter(data: 'np.ndarray', *options, **kwoptions):
    """
    Filters the image to only show the specified color, with an optional scale
    
    Parameters
    ----------
    data : np.ndarray
        The image to filter
    color : list[str], optional
        The colors to filter
    scale : int | float | list[int | float] | dict[str, int | float], optional
        The scale to apply
        Can be a number, a list, or a dictionary of numbers
    """
    # color:list[str], scale:int|float|list[int|float]|dict[str,int|float]=1
    f = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]
    
    if ftype != 'ppm':
        raise ValueError("Data must be a color image")
    
    color, scale = kwoptions.get('color', None), kwoptions.get('scale', None)
    
    if len(options) > 2:
        raise ValueError("Too many options")
        
    """
    I want to do something better for these checks.
    They are pretty bad, and don't even allow the passing
    of a regular dictionary. For now, this will do.
    """
    # This converts options into something compatible with the checks immediately below
    if type(options[0]) == dict:
        if len(options) > 1:
            pass
        else:
            options = list(zip(*options[0].items()))
            if len(options[1]) == 1:
                options[1] = options[1][0]

    if scale is None:
        if options:
            if type(options[-1]) in (int, float) or type(options[-1][0]) in (int, float):
                scale = options[-1]
            elif type(options[0]) in (int, float) or type(options[0][0]) in (int, float):
                scale = options[0]
        # It is an if statement bc scale could still be None
        if not scale:
            scale = 1
    
    if color is None:
        if options:
            if type(options[0][0]) == str:
                color = options[0]
            elif type(options[-1][0]) == str:
                color = options[-1]
        # Same thing here
        if not type(scale) == dict and not color:
            color = ['r', 'g', 'b']
    """End of checks"""

    if set(color) == {'r', 'g', 'b'} and scale == 1:
        return data
    
    if len(color) > 3:
        raise ValueError("Invalid color requests: Too many color requests")
    elif len(color) < 1:
        raise ValueError("Invalid color requests: No color requests")
    
    # Need to make sure that they are valid strings
    if not all(c.isalpha() for c in color):
        raise ValueError("Invalid color requests: Alphabetic characters only")
    
    valid_colors = ['r', 'g', 'b']
    # Only need the first letter, lowercase too
    color = [ c.lower()[0] for c in color ]
    # Need all colors to be valid
    if not all(c in valid_colors for c in color):
        raise ValueError("Invalid color requests: Request not found")
    # No dupes allowed
    if len(color) != len(set(color)):
        raise ValueError("Invalid color requests: Duplicate colors")
    
    try:
        scale = _check_scale(color, scale)
    except ValueError as e:
        # I want the error in this function, not that one
        raise e
    
    new_data = data.astype(np.uint16)
    for i in range(3):
        new_data[..., i] = new_data[..., i] * scale[valid_colors[i]]
    
    # Made a function to keep it simpler
    new_data = _scale_normalize(new_data, np.max(data))

    file_types[id(new_data)] = 'ppm'
    fin_f = t.time()
    if called_from_main(): print("FILTER:", fin_f - f)

    # returns new_data
    return new_data

def _check_scale(color:list, scale:int|list|dict):
    valid_colors = ['r', 'g', 'b']

    if type(scale) in (list, tuple):
        # If scale is a list, then it needs to be the same length as color
        if len(scale) != len(color):
            raise ValueError("Invalid scale requests: Length mismatch")
        # If it is a list, then it needs to be all positive integers
        if not all(type(s) in (int, float) and s >= 0 for s in scale):
            raise ValueError("Invalid scale requests: Must all be positive numbers")
        # It needs to be 3 long, so we need to fill in the blanks
        
        if type(color) == int:
            scale = {
                c: (scale[color.index(c)]
                    if c in color else 1)
                for c in valid_colors
            }
        elif type(color) == list:
            scale = {
                c: (scale[color.index(c)]
                    if c in color else 0)
                for c in valid_colors
            }

    elif type(scale) == dict:
        # If it is a dictionary, then it needs to have all positive integers
        if not all(type(s) in (int, float) and s >= 0 for s in scale.values()):
            raise ValueError("Invalid scale requests: Must all be positive numbers")
        if color is not None and (not all( k in color for k in scale )):
            raise ValueError("Invalid scale requests: All scale keys must be in color list")
        
        # Make new dictionary
        if color is None:
            scale = { c: scale.get(c, 1) for c in valid_colors }
        else:
            scale = {
                c: scale.get(c, 1)
                if c in color else 0
                for c in valid_colors
            }

    elif type(scale) in (int, float):
        # If it is a number, then it needs to be positive
        if scale < 0:
            raise ValueError("Invalid scale requests: Must be a positive number")
        scale = { c: scale if c in color else int(scale!=1) for c in valid_colors }
    else:
        raise ValueError("Invalid scale requests: Must be an number, list, or dictionary")
    return scale

def _scale_normalize(new_data, max_dval):
    max_val = np.max(new_data)
    
    # If the max value is greater than 255, then we need to scale it down
    if np.max(max_val) > 255:
        new_data = (new_data / max_val * max_dval).astype(np.uint8)

    return new_data

# I made this function and the color class, but it is noticably slower for larger files
def complimentary_switch(data: 'np.ndarray'):
    """Switches the complimentary colors of the image"""

    c = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    if ftype != 'ppm':
        raise ValueError("Data must be a color image")
    
    new_data = np.zeros((len(data), len(data[0]), 3), np.uint8)
    for i in range(len(data)):
        for j in range(len(data[0])):
            complimented_rgb = color().shift_hsv(*data[i][j], 180)
            new_data[i][j] = complimented_rgb

    file_types[id(new_data)] = 'ppm'
    fin_c = t.time()
    if called_from_main(): print("COMPLIMENTARY:", fin_c - c)

    return new_data

class color:
    _instance = None

    def __new__(cls):
        """Ensures there is only one instance of this class"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def toHSV(self, rgb:list):
        r, g, b = [c / 255 for c in rgb]
        max_rgb = max(r, g, b)
        min_rgb = min(r, g, b)
        delta = max_rgb - min_rgb

        v = max_rgb

        if delta == 0:
            h = s = 0
            return h, s, v

        s = delta / max_rgb

        if max_rgb == r:
            h = 60 * (((g - b) / delta) % 6)
        elif max_rgb == g:
            h = 60 * (((b - r) / delta) + 2)
        elif max_rgb == b:
            h = 60 * (((r - g) / delta) + 4)

        return h, s, v
    
    def toRGB(self, hsv:list):
        h, s, v = hsv

        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        elif 300 <= h < 360:
            r, g, b = c, 0, x

        r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
        return int(r), int(g), int(b)
    
    def shift_hsv(self, r, g, b, h_shift):
        h, s, v = self.toHSV([r, g, b])

        h = (h + h_shift) % 360 

        return self.toRGB([h, s, v])
    
# AI made this function. It is more than 10x as fast, so I had it made.
def fast_compliment(data: 'np.ndarray'):
    """Switches the complementary colors of the image using vectorized operations"""

    f = t.time()
    # Access types
    global file_types
    ftype = file_types[id(data)]

    if ftype != 'ppm':
        raise ValueError("Data must be a color image")

    # Normalize RGB values to the range [0, 1]
    data = data / 255.0

    # Split the RGB channels
    r, g, b = data[..., 0], data[..., 1], data[..., 2]

    # Calculate max, min, and delta for HSV conversion
    max_rgb = np.maximum.reduce([r, g, b])
    min_rgb = np.minimum.reduce([r, g, b])
    delta = max_rgb - min_rgb

    # Initialize HSV components
    h = np.zeros_like(max_rgb)
    s = np.zeros_like(max_rgb)  # Initialize saturation to zero
    v = max_rgb

    # Calculate saturation (S) only where max_rgb is not zero
    non_zero_mask = max_rgb != 0
    s[non_zero_mask] = delta[non_zero_mask] / max_rgb[non_zero_mask]

    # Calculate hue (H)
    mask_r = (max_rgb == r) & (delta != 0)
    mask_g = (max_rgb == g) & (delta != 0)
    mask_b = (max_rgb == b) & (delta != 0)

    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    h = np.nan_to_num(h)  # Replace NaN values (from division by zero) with 0

    # Shift hue by 180 degrees
    h = (h + 180) % 360

    # Convert HSV back to RGB
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c

    # Initialize RGB channels
    r_new = np.zeros_like(h)
    g_new = np.zeros_like(h)
    b_new = np.zeros_like(h)

    # Assign RGB values based on hue ranges
    mask = (0 <= h) & (h < 60)
    r_new[mask], g_new[mask], b_new[mask] = c[mask], x[mask], 0

    mask = (60 <= h) & (h < 120)
    r_new[mask], g_new[mask], b_new[mask] = x[mask], c[mask], 0

    mask = (120 <= h) & (h < 180)
    r_new[mask], g_new[mask], b_new[mask] = 0, c[mask], x[mask]

    mask = (180 <= h) & (h < 240)
    r_new[mask], g_new[mask], b_new[mask] = 0, x[mask], c[mask]

    mask = (240 <= h) & (h < 300)
    r_new[mask], g_new[mask], b_new[mask] = x[mask], 0, c[mask]

    mask = (300 <= h) & (h < 360)
    r_new[mask], g_new[mask], b_new[mask] = c[mask], 0, x[mask]

    # Combine RGB channels and scale back to [0, 255]
    rgb = np.stack([r_new + m, g_new + m, b_new + m], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    file_types[id(rgb)] = 'ppm'
    fin_f = t.time()
    if called_from_main(): print("COMPLIMENTARY:", fin_f - f)

    return rgb

def combine(data: list | list[list], axis=1) -> list:
    """Combines multiple images together"""

    c = t.time()
    # Access types
    global file_types
    ftype = set()
    
    depth = _list_depth(data)
    
    if depth not in (1, 2):
        raise ValueError("Data must be a 1D or 2D list")
    data = [ list(p) for p in data ] if depth == 2 else list(data)
    """I might make this work for different sized images that eventually equal out over multiple images, but I've 
    spent to much time on this project already. Like an image stack 4 high would have equal height as a stack of 3."""
    # This loop goes over each array in the list and converts it to PPM format
    for i in range(len(data)):
        # If it is a 2D list
        if depth == 2:
            # Loop over the inner list
            for j in range(len(data[i])):
                size = (height(data[i][j]), width(data[i][j]))
                # If it is a binary or gray image, convert it to a ppm format to be combinable
                if file_types[id(data[i][j])] == 'pbm':
                    data[i][j] = to_ppm(data[i][j]) * 255
                    # Add the type to the set, so no duplicates
                    ftype.add('pbm')
                elif file_types[id(data[i][j])] == 'pgm':
                    data[i][j] = to_ppm(data[i][j])
                    ftype.add('pgm')
                else:
                    ftype.add('ppm')

                # Check if all images are the same size
                if (height(data[i][j]), width(data[i][j])) != size:
                    raise ValueError("Data must all be same-sized images")
        # If it is instead 1D
        else:
            size = (height(data[i]), width(data[i]))
            # If it is a binary or gray image, convert it to a ppm format to be combinable
            if file_types[id(data[i])] == 'pbm':
                data[i] = to_ppm(data[i]) * 255
                ftype.add('pbm')
            elif file_types[id(data[i])] == 'pgm':
                data[i] = to_ppm(data[i])
                # Add the type to the set, so no duplicates
                ftype.add('pgm')
            else:
                ftype.add('ppm')
                
            # Check if all images are the same size
            if (height(data[i]), width(data[i])) != size:
                raise ValueError("Data must all be same-sized images")
    
    # Once all the data is compatible, turn the whole thing into an array
    try:
        new_data = np.array(data, dtype=np.uint8)
    except Exception as e:
        raise ValueError("Data must all be same-sized images")
            
    if depth == 2:
        if axis == 1:
            new_data = np.swapaxes(new_data, 0, 1)
        # Make the axis 0 for the next step
        axis = 0

        # Combine all the images on this axis to a single array
        new_data = np.concatenate(new_data, axis=2)
    # Same thing here
    new_data = np.concatenate(new_data, axis=axis)

    # If it is just a pbm file...
    if ftype == {'pbm'}:
        new_data = new_data[..., 0] // 255
        ftype = 'pbm'

    # If it is a pgm file with or without a pbm file...
    elif ftype - {'pbm'} == {'pgm'}:
        new_data = new_data[..., 0]
        ftype = 'pgm'

    # If it includes a ppm file...
    else:
        ftype = 'ppm'
        
    file_types[id(new_data)] = ftype
    fin_c = t.time()
    if called_from_main(): print("COMBINE:", fin_c - c)

    return new_data

def _list_depth(data: list) -> int:
    """Returns the depth of a list"""
    if isinstance(data, (list, tuple)):
        return 1 + _list_depth(data[0])
    else:
        return 0

def to_ppm(data: 'np.ndarray') -> 'np.ndarray':
    """Converts a 2D array to a 3D array with 3 color values"""

    p = t.time()
    global file_types
    ftype = file_types[id(data)]

    if ftype == 'ppm':
        print("Data already in ppm format")
        new_data = data
    else:
        new_data = np.repeat(data[:, :, np.newaxis], 3, axis=2)

    file_types[id(new_data)] = 'ppm'
    fin_p = t.time()
    if called_from_main(): print("TO PPM:", fin_p - p)

    return new_data
    
def display(data: 'np.ndarray', cmap=None):
    """Displays a Numpy array as an image"""
    
    d = t.time()

    try:
        check_data(data)
    except ValueError as e:
        raise e
    
    # Access types
    global file_types
    ftype = file_types[id(data)]

    if not cmap:
        if ftype == 'pgm':
            plt.set_cmap('binary')
        elif ftype == 'pbm':
            plt.set_cmap('gray')

    plt.imshow(data, cmap)
    fin_d = t.time()
    print('DISPLAY:', fin_d - d)
    plt.axis("off")
    plt.show()

def check_data(data: 'np.ndarray', return_shape=False):
    """Checks the data to see if it is a valid image"""
    # Access types
    global file_types
    ftype = file_types[id(data)]

    # If data is not an ndarray, then raise an exception:
    if not isinstance(data, np.ndarray):
        raise ValueError('Data is not a Numpy array.')
    # If data has mismatched lengths, then raise an exception:
    if ftype == 'ppm':
        data_len = len(np.reshape(data, (-1, 3), copy=True))
    else:
        data_len = len(data.flatten())
    if data_len != len(data) * len(data[0]):
        raise ValueError('Data is an invalid shape.')
    # Make sure all color values are valid
    if np.max(data) > 255 or np.min(data) < 0:
        raise ValueError('Data contains invalid color values.')
    if return_shape:
        return data.shape

def file2list(path: str, P: list[str], max_char:int):
    """Clears the list P and fills it with all words from file 'path', ignoring comments"""
    
    try:
        # Open the file:
        with open(path, 'r') as f:
            # Clear the list P:
            P *= 0
            in_comment = False
            # Parse the file one chunk at a time:
            while True:
                chunk, in_comment = read_chunk(f, max_char, in_comment)
                if chunk is None:
                    break
                P += chunk.split()
        # Return True:
        return [True]
    except Exception:
        e = tb.format_exc()
        # Return False
        return [False, e]

# This function actually can have many uses outside this program, with some tweaks
def read_chunk(f, max_char: int, in_comment: bool) -> str:
    chunk:str = f.read(max_char)
    if not chunk:
        return None, in_comment  # End of file

    # Compensate for incomplete words
    if len(chunk) == max_char:
        while not chunk[-1].isspace() and not chunk[-1] == '\n':
            next_char = f.read(1)
            if not next_char:
                break  # End of file
            chunk += next_char

    # Remove comments
    lines = chunk.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if in_comment:
            # Set in_comment to False and skip the line
            if i < len(lines) - 1:
                in_comment = False
                continue
            else:
                # If the comment is the last item, it isn't terminated
                break

        comment_idx = line.find('#')
        if comment_idx != -1:
            new_lines.append(line[:comment_idx])
            # If the comment is the last item, set in_comment to True
            if i == len(lines) - 1:
                in_comment = True
        else:
            new_lines.append(line)
    chunk = '\n'.join(new_lines)

    return chunk.strip(), in_comment

# AI made this for me, but I do understand it now
def called_from_main():
    """Used to time functions called by main program"""
    # Get the current call stack
    stack = inspect.stack()
    # Check the caller's frame
    caller_frame = stack[2]
    # If it is called from the grouped functions, then allow one more layer
    if caller_frame.function[:4] == 'Part':
        caller_frame = stack[3]
    # Get the caller's function name
    caller_function = caller_frame.function
    # If the caller function is '<module>', it was called from the main script
    return caller_function == '<module>'

# I made these for ease of use, so I don't have to always type len(data[0]), for example
def width(data: 'np.ndarray') -> int:
    """
    Returns the width of the array
    """
    return data.shape[1]

def height(data: 'np.ndarray') -> int:
    """
    Returns the height of the array
    """
    return data.shape[0]

##### Do not change code below this line #####

def readfile(path:str, max_char=50_000):
    """Reads Portable Map image from file 'path' and returns it as a Numpy array."""

    # Create an empty list L:
    L = []

    file = t.time()
    # Read file 'path' into the list L:
    success = file2list(path, L, max_char)
    fin_file = t.time()

    if not success[0]:
        raise Exception(f"{str(success[1])}")
    print('FILE:', fin_file - file)
    
    r = t.time()
    # Convert list L to a 3D array 'data':
    data = render(L)
    fin_r = t.time()
    print('RENDER:', fin_r - r)

    file_types[id(data)] = path[-3:]
    # Return 'data':
    return data

# Main program:
import matplotlib.pyplot as plt, numpy as np, traceback as tb, time as t, inspect
file_types = dict()

path = "dog_filter.ppm"

# I changed code beyond the line. Oops \(*O*)/
D = readfile(path)

w = width(D)
h = height(D)

times = ('display', 'render', 'writemapfile', 'hflip', 'vflip', 'rotate', 'shrink', 'tile', 'subimage', 'makegray', 'makebinary', 'filter', 'complimentary_switch', 'fast_compliment', 'combine', 'to_ppm', )

def Part1():
    """Part 1 shows off the rotate() function"""
    # Rotate large blocks around the center by increments of 90 degrees:
    D_rotate1a = rotate(D, 90, [h//2, w//2])
    D_rotate2a = rotate(D, 180, [h//2, w//2])
    D_rotate3a = rotate(D, 270, [h//2, w//2])

    # Rotate tiny blocks around the center by increments of 90 degrees:
    D_rotate1b = rotate(D, 90, [4, 4])
    D_rotate2b = rotate(D, 180, [4, 4])
    D_rotate3b = rotate(D, 270, [4, 4])

    # Rotate large blocks in place by increments of 90 degrees:
    D_rotate4a = rotate(D, 90, [h//2, w//2], True)
    D_rotate5a = rotate(D, 180, [h//2, w//2], True)
    D_rotate6a = rotate(D, 270, [h//2, w//2], True)

    # Rotate tiny blocks in place by increments of 90 degrees:
    D_rotate4b = rotate(D, 90, [4, 4], True)
    D_rotate5b = rotate(D, 180, [4, 4], True)
    D_rotate6b = rotate(D, 270, [4, 4], True)

    data = [D, D_rotate1a, D_rotate2a, D_rotate3a, D, D_rotate1b, D_rotate2b, D_rotate3b, D, D_rotate4a, D_rotate5a, D_rotate6a, D, D_rotate4b, D_rotate5b, D_rotate6b, D]

    for i in data:
        display(i)

# Part1()

def Part2():
    """Part 2 shows off the subimage() function"""
    # Ctrl + click to see the brightness function easily
    brightness
    # I may change how it handles the function arguments, but I'll have to see
    D_sub = subimage(D, [0, 0], [w//2, h//2], brightness=[25])
    D_sub = subimage(D_sub, [w//2, 0], [w, h//2], brightness=[50])
    D_sub = subimage(D_sub, [w//2, h//2], [w, h], brightness=[75])

    display(D_sub)

# Part2()

def Part3():
    """Part 3 shows off most of the other functions"""

    # Shrink the image by a factor of 2
    D_shrink = shrink(D, 2)
    # Tile the image 2 high and 2 wide
    D_tile = tile(D_shrink, 2, 2)
    # Makes the image grayscale
    D_gray = makegray(D)
    # Brighten by 50%
    D_bright = brightness(D, 150)
    # Darken by 50%
    D_dark = brightness(D, 50)
    # Darken the 'g' and 'b' to 85%, but leaves the 'r' untouched (because scale is an integer)
    D_filter_gb = filter(D, ['g', 'b'], 0.5)
    # Scales the 'r' and 'b' by 1.3 and 2, respectively, removing 'g' entirely (because scale is a list with corresponding values)
    D_filter_rb = filter(D, ['r', 'b'], [1.3, 2])
    # Filter out the blue
    D_filter_rg = filter(D, ['r', 'g'])
    # Make the 'r' channel more potent, but leaves the 'g' and 'b' untouched
    D_filter_r = filter(D, {'r': 1.5})
    # Invert the colors' hues
    D_comp = fast_compliment(D)
 
    final_D1 = combine(((D, D_gray, D_tile), (D_bright, D_dark, D_comp), (D_filter_gb, D_filter_rb, D_filter_rg)), axis=1) # axis=1 is default
    final_D2 = combine(((D, D_gray, D_tile), (D_bright, D_dark, D_comp), (D_filter_gb, D_filter_rb, D_filter_rg)), axis=0)
    final_rotate_D = rotate(final_D1, 90, (h, w))
    final_flip_D = hflip(final_D1, w)
    D_only_filter = combine(((D_filter_gb, D_filter_rb), (D_filter_rg, D_filter_r)))

    display(final_flip_D)
    display(final_D1)
    display(final_rotate_D)
    display(final_D2)
    display(D_only_filter)

    writemapfile("dog_filter.ppm", D_only_filter)

# Part3()

if path == "dog_filter.ppm":
    display(D)