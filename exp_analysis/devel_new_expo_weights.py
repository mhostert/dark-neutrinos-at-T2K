import numpy as np
from parameters_dict import tpc_fiducial_volume_endpoints, tpc_length, fgd_length
exp_analysis = 1

def compute_decay_integral(df):
        df['pdark_dir', 'x'] = df['pdark', 'x']/df['p3dark', '']
        df['pdark_dir', 'y'] = df['pdark', 'y']/df['p3dark', '']
        df['pdark_dir', 'z'] = df['pdark', 'z']/df['p3dark', '']
        
        int_point_z = df['int_point', 'z']
        exp_integral_points = np.zeros((len(df), 6))

        t_exit_x_min = (tpc_fiducial_volume_endpoints[0][0] - df['int_point_x'])/df['pdark_dir_x']
        t_exit_x_max = (tpc_fiducial_volume_endpoints[0][1] - df['int_point_x'])/df['pdark_dir_x']
        t_exit_y_min = (tpc_fiducial_volume_endpoints[1][0] - df['int_point_y'])/df['pdark_dir_y']
        t_exit_y_max = (tpc_fiducial_volume_endpoints[1][1] - df['int_point_y'])/df['pdark_dir_y']
        t_exit_s = np.stack([t_exit_x_min, t_exit_x_max, t_exit_y_min, t_exit_y_max], axis=1)

        t_exit_x_min_in = exp_analysis.is_point_in_tpc1d(df['int_point_y'] + t_exit_x_min*df['pdark_dir_y'], coord=1)
        t_exit_x_max_in = exp_analysis.is_point_in_tpc1d(df['int_point_y'] + t_exit_x_max*df['pdark_dir_y'], coord=1)
        t_exit_y_min_in = exp_analysis.is_point_in_tpc1d(df['int_point_x'] + t_exit_y_min*df['pdark_dir_x'], coord=0)
        t_exit_y_max_in = exp_analysis.is_point_in_tpc1d(df['int_point_x'] + t_exit_y_max*df['pdark_dir_x'], coord=0)
        are_t_exit_s_in = np.stack([t_exit_x_min_in, t_exit_x_max_in, t_exit_y_min_in, t_exit_y_max_in], axis=1)

        t_exit_s[~are_t_exit_s_in] = np.inf
        t_exit_s = np.sort(t_exit_s, axis=1)

        for tpc_index in range(3):
            t_tpc_0 = np.clip((tpc_fiducial_volume_endpoints[2][0] + tpc_index * (tpc_outer_volume[0] + fgd_outer_volume[0]) - int_point_z)/df['pdark_dir_z'], a_min=0, a_max=None)
            t_tpc_1 = np.clip((tpc_fiducial_volume_endpoints[2][1] + tpc_index * (tpc_outer_volume[0] + fgd_outer_volume[0]) - int_point_z)/df['pdark_dir_z'], a_min=0, a_max=None)
            t_tpc_s = np.stack([t_tpc_0, t_tpc_1], axis=1)
            t_tpc_s = np.sort(t_tpc_s, axis=1)
            are_t_tpc_s_in = exp_analysis.is_point_in_tpc2d(np.expand_dims(df['int_point_x'], [1]) + np.expand_dims(df['pdark_dir_x'], [1])*t_tpc_s,
                                                np.expand_dims(df['int_point_y'], [1]) + np.expand_dims(df['pdark_dir_y'], [1])*t_tpc_s)

            # this line is problematic
            t_exit = t_exit_s[(t_exit_s>t_0)&(t_exit_s<t_1)]
            
            condlist = [
                is_t_0_in & is_t_1_in, 
                (~is_t_0_in) & (~is_t_1_in),
                is_t_0_in & (~is_t_1_in),
                (~is_t_0_in) & is_t_1_in,
                        ]
            choicelist_in = [
                t_0,
                0,
                t_0,
                t_exit
            ]
            choicelist_out = [
                t_1,
                0,
                t_exit,
                t_1
            ]
            exp_integral_points[:,tpc_index*2] = np.select(condlist, choicelist_in)
            exp_integral_points[:,tpc_index*2+1] = np.select(condlist, choicelist_out)

        for i in range(6):
            df[f'exp_integral_points_{i}'] = exp_integral_points[:,i]

        how_many_in = are_t_exit_s_in.sum(axis=1)
        if how_many_in == 1 or how_many_in == 3:
            for i in range(6):
                df[f'exp_integral_points_{i}'] = 0