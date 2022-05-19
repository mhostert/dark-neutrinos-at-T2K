def compute_decay_integral(df):
        df['pdark_dir', 'x'] = df['pdark', 'x']/df['p3dark', '']
        df['pdark_dir', 'y'] = df['pdark', 'y']/df['p3dark', '']
        df['pdark_dir', 'z'] = df['pdark', 'z']/df['p3dark', '']
        
        int_point_z = df['int_point', 'z']
        exp_integral_points = np.zeros((len(df), 6))

        poe_x_min = (tpc_fiducial_volume_endpoints[0][0] - df['int_point', 'x'])/df['pdark_dir', 'x']
        poe_x_max = (tpc_fiducial_volume_endpoints[0][1] - df['int_point', 'x'])/df['pdark_dir', 'x']
        poe_y_min = (tpc_fiducial_volume_endpoints[1][0] - df['int_point', 'y'])/df['pdark_dir', 'y']
        poe_y_max = (tpc_fiducial_volume_endpoints[1][1] - df['int_point', 'y'])/df['pdark_dir', 'y']
        poe_s = np.stack([poe_x_min, poe_x_max, poe_y_min, poe_y_max], axis=1)
        are_poe_s_in = exp_analysis.is_point_in_tpc2(df['int_point', 'x'] + df['pdark_dir', 'x']*poe_s,
                                                     df['int_point', 'y'] + df['pdark_dir', 'y']*poe_s)
        poe_s[~are_poe_s_in] = np.inf

        for tpc_index in range(3):
            t_0 = np.clip((tpc_fiducial_volume_endpoints[2][0] + tpc_index * (tpc_length + fgd_length) - int_point_z)/df['pdark_dir', 'z'], a_min=0)
            t_1 = np.clip((tpc_fiducial_volume_endpoints[2][1] + tpc_index * (tpc_length + fgd_length) - int_point_z)/df['pdark_dir', 'z'], a_min=0)
            is_t_0_in = exp_analysis.is_point_in_tpc2(df['int_point', 'x'] + df['pdark_dir', 'x']*t_0,
                              df['int_point', 'y'] + df['pdark_dir', 'y']*t_0)
            is_t_1_in = exp_analysis.is_point_in_tpc2(df['int_point', 'x'] + df['pdark_dir', 'x']*t_1,
                              df['int_point', 'y'] + df['pdark_dir', 'y']*t_1)

            # this line is problematic
            poe = poe_s[(poe_s>t_0)&(poe_s<t_1)]
            
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
                poe
            ]
            choicelist_out = [
                t_1,
                0,
                poe,
                t_1
            ]
            exp_integral_points[:,tpc_index*2] = np.select(condlist, choicelist_in)
            exp_integral_points[:,tpc_index*2+1] = np.select(condlist, choicelist_out)

        for i in range(6):
            df[f'exp_integral_points_{i}'] = exp_integral_points[:,i]