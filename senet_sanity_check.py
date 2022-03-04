__data_source,__filelist,__identities = __create_filelists(__train_folder)
__identities = __identities[:2]
n=2
__filelist,__identities = __only_keep_n_classes(__filelist,__identities,n)
__mini_dataset = VGGFACE2_100(__data_source,__filelist,__identities)
batch_size = 32
__loader = torch.utils.data.DataLoader(__what__,batch_size=batch_size,shuffle=False,num_workers=4)

#---------------------------------------------------------------------------------------------------------------------------------
__obj = __write_all_embeddings_to_disk(net,__loader)
__other_data = None
__retrieve_with_mean(__obj,__other_data)
# hmm 2 classes might not be enough to address mistakes...
#TODO we need to check the mistaken identities within the retrieval function itself..
#TODO find other reasons for retrieval failure
#---------------------------------------------------------------------------------------------------------------------------------
it = iter(__loader)

#---------------------------------------------------------------------------------------------------------------------------------
embeddings,classes = [],[]
for bx,by in it:
    bx = bx.to(device)
    net = net.to(device)
    by = by.to(device)

    b_embeddings = net(bx)
    embeddings.append(b_embeddings)
    classes.append(b_embeddings)

embeddings = torch.stack(embeddings,0)
classes = torch.stack(classes,0)
uq_classes = torch.unique(classes)


#---------------------------------------------------------------------------------------------------------------------------------
means = {}
where = {}
for c in uq_classes:
    where_c = classes == c
    where[c] = where_c
    embeddings_of_c = embeddings[where_c]
    mean_c = torch.mean(embeddings_of_c,0)
    means[c] = mean_c


#---------------------------------------------------------------------------------------------------------------------------------
distances = {}
for ref_c in uq_classes:
    mean_ref_c = means[ref_c]
    for other_c in uq_classes:
        embeddings_other_c = embeddings[where[other_c]]
        distances[(ref_c,other_c)] = __pdist(embeddings_other_c,mean_c)


#---------------------------------------------------------------------------------------------------------------------------------

__d_thresh_range = [1.,2.]
for d in __d_thresh_range:
# can we just run retrieval from mean here?

    
