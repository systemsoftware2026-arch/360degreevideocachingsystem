import roi_info
import random
import numpy as np
def viewportDataGenerator(num_tile_per_seg,tile_popularity,num_bandwidth_class):

    random.seed(10)
    np.random.seed(10)
    num_request_per_seg=num_bandwidth_class
    ret_vp_tiles=[]
    tiles_p=np.zeros(num_tile_per_seg)
    for tile in range(num_tile_per_seg):
        if(tile==0):
            tiles_p[tile]=tile_popularity[tile]
        else:
            tiles_p[tile] = tile_popularity[tile]+tiles_p[tile-1]
    for i in range(num_request_per_seg):
        vp_tiles = []
        vp_rand_p = random.uniform(0,1)
        vp_center=0
        for tile_idx in range(num_tile_per_seg):
            if(vp_rand_p<=tiles_p[tile_idx]):
                vp_center=tile_idx
                vp_tiles = roi_info.generate_viewport(vp_center)
                break


        ret_vp_tiles.append(vp_tiles)


    return ret_vp_tiles

