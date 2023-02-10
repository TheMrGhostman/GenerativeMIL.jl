using Flux3D

function simple_experiment_evaluation(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	#temporary function
	tr_data, val_data, tst_data = data
	# unpack data for easier manipulation
	tr_data, tr_lab = Models.unpack_mill(tr_data)
	val_data, val_lab = Models.unpack_mill(val_data)
	tst_data, tst_lab = Models.unpack_mill(tst_data)

	# create reconstructed bags
	tr_rec = cat(score_fun(tr_data)..., dims=3)
	val_rec= cat(score_fun(val_data)..., dims=3)
	tst_rec= cat(score_fun(tst_data)..., dims=3)

	tr_score = Flux3D.chamfer_distance(tr_data, tr_rec)
	val_score = Flux3D.chamfer_distance(val_data, val_rec)
	tst_score = Flux3D.chamfer_distance(tst_data, tst_rec)

	savef = joinpath(savepath, savename(merge(parameters, (type = "reconstructed_input",)), "bson", digits=5))
	result = (
		parameters = merge(parameters, (type = "reconstructed_input",)),
		loss_train = tr_score,
		loss_valid = val_score, 
		loss_test  = tst_score,
		)
	result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
	if save_result
		tagsave(savef, result, safe = true)
		verb ? (@info "Results saved to $savef") : nothing
	end
	return result
end