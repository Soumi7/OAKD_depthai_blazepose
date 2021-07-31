"""
This file is the template of the scripting node source code in edge mode
Substitution is made in BlazeposeDepthaiEdge.py

In the following:
rrn_ : normalized [0:1] coordinates in rotated rectangle coordinate systems
sqn_ : normalized [0:1] coordinates in squared input image 
"""

${_TRACE} ("Starting manager script node")

import marshal
from math import sin, cos, atan2, pi, hypot, degrees, floor

# Indexes of some keypoints 
left_shoulder = 11
right_shoulder = 12
left_hip = 23
right_hip = 24

${_IF_XYZ}
# We use a filter for smoothing the reference point (mid hips or mid shoulders)
#  for which we fetch the (x,y,z). Without this filter, the reference point is very shaky
class SmoothingFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.initialized = False
    def apply(self, x):
        if self.initialized:
            result = self.alpha * x + (1 - self.alpha) * self.prev_x
        else:
            result = x
            self.initialized = True
        self.prev_x = result
        return int(result)
    def reset(self):
        self.initialized = False

filter_x = SmoothingFilter(0.5)
filter_y = SmoothingFilter(0.5)
${_IF_XYZ}

def send_result(buf, type, lm_score=0, rect_center_x=0, rect_center_y=0, rect_size=0, rotation=0, lms=0, lms_world=0, xyz_ref=0, xyz=0, xyz_zone=0):
    # type : 0, 1 or 2
    #   0 : pose detection only (detection score < threshold)
    #   1 : pose detection + landmark regression
    #   2 : landmark regression only (ROI computed from previous landmarks)
    result = dict([("type", type), ("lm_score", lm_score), ("rotation", rotation),
            ("rect_center_x", rect_center_x), ("rect_center_y", rect_center_y), ("rect_size", rect_size), 
            ("lms", lms), ('lms_world', lms_world),
            ("xyz_ref", xyz_ref), ("xyz", xyz), ("xyz_zone", xyz_zone)])
    result_serial = marshal.dumps(result, 2)
    ${_TRACE} ("len result:"+str(len(result_serial)))
    
    buf.getData()[:] = result_serial  
    node.io['host'].send(buf)
    ${_TRACE} ("Manager sent result to host")


def rr2img(rrn_x, rrn_y): 
    # Convert a point (rrn_x, rrn_y) expressed in normalized rotated rectangle (rrn)
    # into (X, Y) expressed in normalized image (sqn)
    X = sqn_rr_center_x + sqn_rr_size * ((rrn_x - 0.5) * cos_rot + (0.5 - rrn_y) * sin_rot)
    Y = sqn_rr_center_y + sqn_rr_size * ((rrn_y - 0.5) * cos_rot + (rrn_x - 0.5) * sin_rot)
    return X, Y

norm_pad_size = ${_pad_h} / ${_frame_size}

def is_visible(lm_id):
    # Is the landmark lm_id is visible ?
    # Here visibility means inferred visibility from the landmark model
    return lms[lm_id*5+3] > ${_visibility_threshold} 

def is_in_image(sqn_x, sqn_y):
    # Is the point (sqn_x, sqn_y) is included in the useful part of the image (excluding the pads)?
    return  norm_pad_size < sqn_y < 1 - norm_pad_size

# send_new_frame_to_branch defines on which branch new incoming frames are sent
# 1 = pose detection branch 
# 2 = landmark branch
send_new_frame_to_branch = 1

# Predefined buffer variables used for sending result to host
buf1 = Buffer(197)
buf2 = Buffer(${_buffer_size})
buf3 = Buffer(201)

next_roi_lm_idx = 33*5

cfg_pre_pd = ImageManipConfig()
cfg_pre_pd.setResizeThumbnail(224, 224, 0, 0, 0)

while True:
    if send_new_frame_to_branch == 1: # Routing frame to pd
        node.io['pre_pd_manip_cfg'].send(cfg_pre_pd)
        ${_TRACE} ("Manager sent thumbnail config to pre_pd manip")
        # Wait for pd post processing's result 
        detection = node.io['from_post_pd_nn'].get().getLayerFp16("result")
        ${_TRACE} ("Manager received pd result: "+str(detection))
        pd_score, sqn_rr_center_x, sqn_rr_center_y, sqn_scale_x, sqn_scale_y = detection
        if pd_score < ${_pd_score_thresh}:
            send_result(buf1, 0)
            continue
        scale_center_x = sqn_scale_x - sqn_rr_center_x
        scale_center_y = sqn_scale_y - sqn_rr_center_y
        sqn_rr_size = 2 * ${_rect_transf_scale} * hypot(scale_center_x, scale_center_y)
        rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
        rotation = rotation - 2 * pi *floor((rotation + pi) / (2 * pi))
        ${_IF_XYZ}
        filter_x.reset()
        filter_y.reset()
        ${_IF_XYZ}
        

    # Routing frame to lm
    sin_rot = sin(rotation) # We will often need these values later
    cos_rot = cos(rotation)
    # Tell pre_lm_manip how to crop body region 
    rr = RotatedRect()
    rr.center.x    = sqn_rr_center_x
    rr.center.y    = (sqn_rr_center_y * ${_frame_size} - ${_pad_h}) / ${_img_h}
    rr.size.width  = sqn_rr_size
    rr.size.height = sqn_rr_size * ${_frame_size} / ${_img_h}
    rr.angle       = degrees(rotation)
    cfg = ImageManipConfig()
    cfg.setCropRotatedRect(rr, True)
    cfg.setResize(256, 256)
    node.io['pre_lm_manip_cfg'].send(cfg)
    ${_TRACE} ("Manager sent config to pre_lm manip")

    # Wait for lm's result
    lm_result = node.io['from_lm_nn'].get()
    ${_TRACE} ("Manager received result from lm nn")
    lm_score = lm_result.getLayerFp16("Identity_1")[0]
    if lm_score > ${_lm_score_thresh}:
        lms = lm_result.getLayerFp16("Identity")
        lms_world = lm_result.getLayerFp16("Identity_4")[:99]
        
        xyz = 0
        xyz_zone = 0
        xyz_ref = 0
        # Query xyz
        ${_IF_XYZ}
        # Choosing the reference point: mid hips if hips visible, or mid shoulders otherwise
        # xyz_ref codes the reference point, 1 if mid hips, 2 if mid shoulders, 0 if no reference point
        if is_visible(right_hip) and is_visible(left_hip):      
            kp1 = right_hip
            kp2 = left_hip
            rrn_xyz_ref_x = (lms[5*kp1] + lms[5*kp2]) / 512 # 512 = 256*2 (256 to normalizing, 2 for the mean)
            rrn_xyz_ref_y = (lms[5*kp1+1] + lms[5*kp2+1]) / 512
            sqn_xyz_ref_x, sqn_xyz_ref_y = rr2img(rrn_xyz_ref_x, rrn_xyz_ref_y) 
            if is_in_image(sqn_xyz_ref_x, sqn_xyz_ref_y):
                xyz_ref = 1
        if xyz_ref == 0 and is_visible(right_shoulder) and is_visible(left_shoulder):
            xyz_ref = 2
            kp1 = right_shoulder
            kp2 = left_shoulder
            rrn_xyz_ref_x = (lms[5*kp1] + lms[5*kp2]) / 512 # 512 = 256*2 (256 to normalizing, 2 for the mean)
            rrn_xyz_ref_y = (lms[5*kp1+1] + lms[5*kp2+1]) / 512
            sqn_xyz_ref_x, sqn_xyz_ref_y = rr2img(rrn_xyz_ref_x, rrn_xyz_ref_y) 
            if is_in_image(sqn_xyz_ref_x, sqn_xyz_ref_y):
                xyz_ref = 2
        if xyz_ref:
            cfg = SpatialLocationCalculatorConfig()
            conf_data = SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 10000
            zone_size = max(int(sqn_rr_size * ${_frame_size} / 45), 8)
            xyz_ref_x = filter_x.apply(sqn_xyz_ref_x * ${_frame_size} -zone_size/2 + ${_crop_w})
            xyz_ref_y = filter_y.apply(sqn_xyz_ref_y * ${_frame_size} -zone_size/2 - ${_pad_h})
            rect_center = Point2f(xyz_ref_x, xyz_ref_y)
            rect_size = Size2f(zone_size, zone_size)
            conf_data.roi = Rect(rect_center, rect_size)
            cfg = SpatialLocationCalculatorConfig()
            cfg.addROI(conf_data)
            node.io['spatial_location_config'].send(cfg)
            ${_TRACE} ("Manager sent ROI to spatial_location_config")
            # Wait xyz response
            xyz_data = node.io['spatial_data'].get().getSpatialLocations()
            ${_TRACE} ("Manager received spatial_location")
            coords = xyz_data[0].spatialCoordinates
            xyz = [float(coords.x), float(coords.y), float(coords.z)]
            roi = xyz_data[0].config.roi
            xyz_zone = [int(roi.topLeft().x - ${_crop_w}), int(roi.topLeft().y), int(roi.bottomRight().x - ${_crop_w}), int(roi.bottomRight().y)]
        else:
            xyz = [0.0] * 3
            xyz_zone = [0] * 4
        ${_IF_XYZ}

        # Send result to host
        send_result(buf2, send_new_frame_to_branch, lm_score, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation, lms, lms_world, xyz_ref, xyz, xyz_zone)
        
        if not ${_force_detection}:
            send_new_frame_to_branch = 2 
            # Calculate the ROI for next frame     
            rrn_rr_center_x = lms[next_roi_lm_idx] / 256
            rrn_rr_center_y = lms[next_roi_lm_idx+1] / 256
            rrn_scale_x = lms[next_roi_lm_idx+5] / 256
            rrn_scale_y = lms[next_roi_lm_idx+6] / 256
            sqn_scale_x, sqn_scale_y = rr2img(rrn_scale_x, rrn_scale_y)
            sqn_rr_center_x, sqn_rr_center_y = rr2img(rrn_rr_center_x, rrn_rr_center_y)
            scale_center_x = sqn_scale_x - sqn_rr_center_x
            scale_center_y = sqn_scale_y - sqn_rr_center_y
            sqn_rr_size = 2 * ${_rect_transf_scale} * hypot(scale_center_x, scale_center_y) 
            rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
            rotation = rotation - 2 * pi *floor((rotation + pi) / (2 * pi))    

    else:
        send_result(buf3, send_new_frame_to_branch, lm_score)
        send_new_frame_to_branch = 1