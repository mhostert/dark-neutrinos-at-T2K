import darkcast
import matplotlib.pyplot as pyplot

name  = 'dark_photon'
model = darkcast.Model(name)
limits = darkcast.Limits()

# # Try to load matplotlib.
# color = "lightgrey"

# # If possible, initialize the plot.
# fig, ax = pyplot.subplots()

# # Loop over the limits.
# for label, limit in limits.items():
#     if limit.model.width('invisible', 1) != 0: continue
#     else: print(label)
    
#     # Recast the limit, this returns an object of type 'Datasets'.
#     recast = limit.recast(model)

#     # Save the limit to a text file. This is done with the
#     # 'Datasets.write' method.
#     recast.write("darkcast/recast/limits/%s/%s.lmt" % (name, label))
#     if pyplot:
#         for x, y in recast.plots():
#             lbl = darkcast.utils.latex(limit)
#             ax.fill(x, y, alpha = 0.95, fill = True, facecolor = 'lightgrey', edgecolor='black', linewidth=0.0)

# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim([2e-3, 1e2])
# ax.set_ylim([1e-8, 1e1])
# fig.savefig("plots/visible_%s.pdf" % name)


def light_dark_photon_limits(ax):
    for limit in limits.values():
        if limit.model.width('invisible', 1) != 0: 
            continue
        recast = limit.recast(model)
        for x, y in recast.plots():
            ax.fill(x, y, alpha = 0.95, 
                    fill=True, color='lightgrey', 
                    edgecolor='black', linewidth=0.0,
                   zorder=0)