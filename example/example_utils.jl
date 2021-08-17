function mycb(p,l,ŷ,y;doplot=true,bids=[1],fids=[1,2])
  display(l)
  if doplot
    y = ndims(y) < 3 ? Flux.stack(y,2) : y
    ŷ = ndims(ŷ) < 3 ? Flux.stack(ŷ,2) : ŷ

    fig = plot(y[1,:,1], label="y1")
    size(y,1) > 1 && plot!(fig, y[2,:,1], label="y2")
    plot!(fig, ŷ[1,:,1], label="ŷ1")
    size(y,1) > 1 && plot!(fig, ŷ[2,:,1], label="ŷ2")
    display(fig)
  end
  return false
end


function plot_wiring(wiring::Wiring)
  display(heatmap(wiring.sens_mask))
  display(heatmap(wiring.sens_pol))
  display(heatmap(wiring.syn_mask))
  display(heatmap(wiring.syn_pol))
end
