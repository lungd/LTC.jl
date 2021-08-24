function mycb(p,l,ŷ,y;doplot=true)
  display(l)
  if doplot
    y = ndims(y) < 3 ? Flux.stack(y,2) : y
    ŷ = ndims(ŷ) < 3 ? Flux.stack(ŷ,2) : ŷ

    if size(y,1) > 3
      hy = heatmap(1:size(y,2),1:size(y,1),y[:,:,1],title="data")
      hŷ = heatmap(1:size(ŷ,2),1:size(ŷ,1),ŷ[:,:,1],title="prediction")
      he = heatmap(1:size(y,2),1:size(y,1),y[:,:,1].-ŷ[:,:,1],title="error")
      p = plot(y[1,:,1], label="y[1,:,1]")
      plot!(p, ŷ[1,:,1], label="ŷ[1,:,1]")
      fig = plot(hy,he,hŷ,p)
      display(fig)
    else
      p = plot(y[1,:,1], label="y1")
      plot!(p, ŷ[1,:,1], label="ŷ1")
      for i in 2:size(y,1)
        plot!(p, y[i,:,1], label="y$(i)")
        plot!(p, ŷ[i,:,1], label="ŷ$(i)")
      end
      display(p)
    end

    # fig = plot(y[1,:,1], label="y1")
    # size(y,1) > 1 && plot!(fig, y[2,:,1], label="y2")
    # plot!(fig, ŷ[1,:,1], label="ŷ1")
    # size(y,1) > 1 && plot!(fig, ŷ[2,:,1], label="ŷ2")
    # display(fig)
    #
    # fig = plot(y[end-1,:,1], label="ye1")
    # size(y,1) > 1 && plot!(fig, y[end,:,1], label="ye2")
    # plot!(fig, ŷ[end-1,:,1], label="ŷe1")
    # size(y,1) > 1 && plot!(fig, ŷ[end,:,1], label="ŷe2")
    # display(fig)
  end
  return false
end


function plot_wiring(wiring::Wiring)
  display(heatmap(wiring.sens_mask))
  display(heatmap(wiring.sens_pol))
  display(heatmap(wiring.syn_mask))
  display(heatmap(wiring.syn_pol))
end
