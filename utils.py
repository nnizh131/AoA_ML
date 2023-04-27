from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def get_data_per_anchor(df, anchor):
    return df.loc[df['anchor'] == anchor]

def spatial_plot(errors, location_x, location_y, filename, title, testing_room='testbench_01_furniture_mid_concrete', mode='xy', vmin=None, vmax=None, cmap='PuBu', anchors=[1,2,3,4]):

    """
    Error per point plot
    """
    errors = pd.DataFrame({

    'coord_x': location_x,

    'coord_y':location_y,

    'errors': errors,

    })
    # errors['xy'] = preds- true_pos
    # errors[['xy']] = true_pos
    # errors['xy'] = mean_absolute_error(preds, true_pos)
    # errors[['x_tag', 'y_tag', 'z_tag']] = true_pos
    ax = plt.gca()
    mappable = errors.plot.hexbin('coord_x', 'coord_y', 'errors', gridsize=(35,12), figsize = (17,7), cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
    addFurniture(ax, testing_room,anchors)
    ax.set_xlabel('x(m)',fontsize=20)
    ax.set_ylabel('y(m)',fontsize=20)
    ax.set_ylim(43,50.3)
    ax.set_xlim(43.9,58.2)
    ax.set_xticklabels(list(range(-2,15,2)))
    ax.set_yticklabels(list(range(0,8)))
    ax.text(60.3, 46, 'MAE (angle)', rotation=90, fontsize=20)
    ax.set_title(title,fontsize=20)
    plt.savefig(filename+'.pdf',bbox_inches='tight', pad_inches = 0)
    plt.show()

def addFurniture(ax, room, anchors=[1,2,3,4]):

    """
    Adds furniture in spatial plot
    """

    a = np.array([1,1,1,1,1,1])
    a_low = 0.5
    if room in ['testbench_01_furniture_mid', 'testbench_01_furniture_mid_concrete']:
        a[1] = a[3] = a_low
    if room in ['testbench_01_furniture_low', 'testbench_01_furniture_low_concrete']:
        a[2] = a[0] = a_low
    if room in ['testbench_01', 'testbench_01_scenario2', 'testbench_01_scenario3']:
        a = 6*[a_low]
    furniture = [plt.Rectangle((44.+1.9, 43.1+0.2), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[0]),
                 plt.Rectangle((44.+4.45, 43.1+1), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[1]),
                 plt.Rectangle((44.+6.4, 43.1+2.6), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[2]),
                 plt.Rectangle((44.+1.7, 43.1+4.1), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[3]),
                 plt.Rectangle((44.+4.2, 43.1+3.4), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[4]),
                 plt.Rectangle((44.+5.4, 43.1+5.15), 0.5, 1, fc='orange', ec='black', lw=2, alpha=a[5]),]
    
    
    anchrs = [plt.Circle((57.9, 43.3), 0.2, fc='firebrick', ec='black', lw=2),
               plt.Circle((57.9, 50.0), 0.2, fc='firebrick', ec='black', lw=2),
               plt.Circle((44.3, 50.0), 0.2, fc='firebrick', ec='black', lw=2),
               plt.Circle((44.3, 43.3), 0.2, fc='firebrick', ec='black', lw=2)]

    for anchor in anchors:
        ax.add_patch(anchrs[anchor-1])
    for item in furniture:
        ax.add_patch(item)
        

def plot_rssi(data, room, channel, anchor_i=0, axes=None, pos=None, vmin=None, vmax=None):

    """
    Spatial visualization for RSSI 
    """

    ax = axes[pos//2,pos%2]
    anchor = 'anchor'+str(anchor_i+1)
    rssi_map = data[room][anchor][channel]['H'].pivot('x_tag', 'y_tag', 'relative_power').values.T[::-1,:]
    rssi_map += data[room][anchor][channel]['H'].pivot('x_tag', 'y_tag', 'reference_power').values.T[::-1,:]
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(rssi_map, cmap='viridis', alpha=0.9, extent=[43.9,58.2, 43,50.3], vmin=vmin, vmax=vmax)
    plt.grid(False)
    addFurniture(ax, room, [anchor_i+1])
    a = ax.get_yticks()
    ax.set_yticks(np.concatenate((a,[a[-1]+1])))
    ax.set_xticks(ax.get_xticks()+2)
    ax.set_xticklabels(list(range(0,15,2)))
    ax.set_yticklabels(list(range(0,8)))
    if pos//2 == 0:
        ax.set_xticks([])
    if pos%2 == 1:
        ax.set_yticks([])
    ax.set_ylim(43,50.4)
    ax.set_xlim(44,58.2)
    ax.grid(False)
    return im


def generalization_plotXY(model_preds, model_names, true_y, training_room='testbench_01_furniture_low', testing_rooms=['testbench_01_furniture_high', 'testbench_01_scenario2', 'testbench_01_scenario3'], mode='xy', pdda=None):
    width=1/len(model_names)
    colors = list(colors.mcolors.TABLEAU_COLORS.values())[:4]
    colors[-1] = list(colors.mcolors.TABLEAU_COLORS.values())[4]

    plt.rcParams["figure.figsize"] = (12,6)

    for i, (model_name, predictions) in enumerate(zip(model_names, model_preds)):
        
        preds = []
        if training_room in predictions[training_room]:
            preds.append((predictions[training_room][training_room] - true_y)**2)
        else:
            preds.append((predictions[training_room] - true_y)**2)
        for testing_room in testing_rooms:
            if testing_room in predictions[training_room]:
                preds.append((predictions[training_room][testing_room] - true_y)**2)
            else:
                preds.append((predictions[testing_room] - true_y)**2)

        maes = [np.mean(np.sqrt(ps[:,0] + ps[:,1])) for ps in preds]   
    
        xpos = np.array(range(len(maes)))
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.grid(b=True, color='gray', axis='y', linestyle='-')
        ax.bar(xpos*width + i, maes, width=width*0.8, color=colors)
    color_name = "gray"
    ax.spines["top"].set_color(color_name)
    ax.spines["bottom"].set_color(color_name)
    ax.spines["left"].set_color(color_name)
    ax.spines["right"].set_color(color_name)
    if pdda:
        er = np.mean(np.sqrt(np.sum(((pdda[testing_rooms[0]] - true_y)**2)[:,:2], axis=1)))
        plt.axhline(y = er, color = 'r', linestyle = '-', label='PDDA')

    plt.xticks(np.arange(len(model_names))+width*0.8*(len(maes)//2)+(len(maes)%2)*width*0.4-width*0.2, model_names)
    
    rooms_dict = {}
    rooms_dict['testbench_01'] = 'No Furniture'
    rooms_dict['testbench_01_furniture_low'] = 'Low Furniture'
    rooms_dict['testbench_01_furniture_high'] = 'High Furniture'
    rooms_dict['testbench_01_scenario2'] = 'Rotated Anchors'
    rooms_dict['testbench_01_scenario3'] = 'Translated Anchors'

    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(1+len(testing_rooms))]
    labels = [rooms_dict[training_room] + ' (Training Room)'] + [rooms_dict[room] for room in testing_rooms]

    if pdda:
        handles.append(plt.Line2D([0], [0], label='PDDA', color='r'))
        labels.append('PDDA')

    plt.legend(handles, labels)
    plt.ylabel('MEDE (m)')

    if mode == 'xy':
        plt.title('XY Error')
    elif mode == 'xyz':
        plt.title('XYZ Error')
        
    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]