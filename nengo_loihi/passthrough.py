import nengo
from nengo.exceptions import BuildError
import numpy as np


def is_passthrough(obj):
    return isinstance(obj, nengo.Node) and obj.output is None


def base_obj(obj):
    if isinstance(obj, nengo.ensemble.Neurons):
        return obj.ensemble
    else:
        return obj


class CycleException(Exception):
    pass


class Cluster(object):
    def __init__(self, obj):
        self.objs = set([obj])
        self.conns_in = set()
        self.conns_out = set()
        self.conns_mid = set()
        self.outputs = {}
        self.probed_objs = set()
    def merge_with(self, other):
        self.objs.update(other.objs)
        self.conns_in.update(other.conns_in)
        self.conns_out.update(other.conns_out)
        self.conns_mid.update(other.conns_mid)
        self.probed_objs.update(other.probed_objs)
        
    def merge_transforms(self, size1, trans1, slice1, node, slice2, trans2, size2):
        if len(trans1.shape) == 0:
            trans1 = np.eye(size1)*trans1   # scalar
        elif len(trans1.shape) != 2:
            raise BuildError('Unhandled transform shape: %s' % trans1.shape)
            
        if len(trans2.shape) == 0:
            trans2 = np.eye(size2)*trans2   # scalar
        elif len(trans2.shape) != 2:
            raise BuildError('Unhandled transform shape: %s' % trans2.shape)
            
        mid_t = np.eye(node.size_in)
        mid_t = mid_t[slice2,slice1]
        t = np.dot(trans2, np.dot(mid_t, trans1))
        return t
        
    def merge_synapses(self, syn1, syn2):
        if syn1 is None:
            return syn2
        elif syn2 is None:
            return syn1
        else:
            return syn1.combine(syn2)

    def generate_from(self, obj, outputs, previous=[]):
        if obj not in outputs:
            return

        if obj in self.probed_objs:
            yield slice(None), np.array(1.0), None, obj

        for c in outputs[obj]:
            if c in self.conns_out:
                yield c.pre_slice, c.transform, c.synapse, c.post
            elif c.post_obj in previous:
                raise CycleException('no loops allowed')
            else:
                for pre_slice, transform, synapse, post in self.generate_from(
                    c.post_obj, outputs, previous=previous+[obj]):
                        
                    syn = self.merge_synapses(c.synapse, synapse)                        
                    trans = self.merge_transforms(c.pre.size_out, c.transform, c.post_slice,
                                                  c.post_obj, pre_slice,
                                                  transform, post.size_in)

                    yield c.pre_slice, trans, syn, post
        
        
        
    def generate_conns(self):
        outputs = {}
        for c in self.conns_in | self.conns_mid | self.conns_out:
            pre = c.pre_obj
            if pre not in outputs:
                outputs[pre] = set([c])
            else:
                outputs[pre].add(c)
        
        
        for c in self.conns_in:
            assert c.post_obj in self.objs
            for pre_slice, transform, synapse, post in self.generate_from(c.post_obj,
                                                               outputs):
                if c.synapse is None:
                    syn = synapse
                elif synapse is None:
                    syn = c.synapse
                else:
                    syn = c.synapse.combine(synapse)
                trans = self.merge_transforms(c.size_mid, c.transform, c.post_slice,
                                              c.post_obj, pre_slice,
                                              transform, post.size_in)
                                              
                if not np.allclose(trans, np.zeros_like(trans)):
                    yield nengo.Connection(
                        pre=c.pre,
                        post=post,
                        function=c.function,
                        eval_points=c.eval_points,
                        scale_eval_points=c.scale_eval_points,
                        synapse=syn,
                        transform=trans,
                        add_to_container=False)
                    
def find_clusters(net, offchip):
    probed_objs = set()
    for p in net.all_probes:
        probed_objs.add(base_obj(p.target))
    clusters = {}
    for c in net.all_connections:
        base_pre = base_obj(c.pre_obj)
        base_post = base_obj(c.post_obj)
        
        pass_pre = is_passthrough(c.pre_obj) and c.pre_obj not in offchip
        if pass_pre and c.pre_obj not in clusters:
            clusters[c.pre_obj] = Cluster(c.pre_obj)
            if c.pre_obj in probed_objs:
                clusters[c.pre_obj].probed_objs.add(c.pre_obj)
            
        pass_post = is_passthrough(c.post_obj) and c.post_obj not in offchip
        if pass_post and c.post_obj not in clusters:
            clusters[c.post_obj] = Cluster(c.post_obj)
            if c.post_obj in probed_objs:
                clusters[c.post_obj].probed_objs.add(c.post_obj)
            
        if pass_pre:
            if pass_post:
                cluster = clusters[base_pre]
                cluster.merge_with(clusters[base_post])
                for obj in cluster.objs:
                    clusters[obj] = cluster
                cluster.conns_mid.add(c)
            else:
                cluster = clusters[base_pre]
                cluster.conns_out.add(c)
        else:
            if pass_post:
                cluster = clusters[base_post]
                cluster.conns_in.add(c)
    return clusters


def convert_passthroughs(network, offchip):
    clusters = find_clusters(network, offchip=offchip)
    
    removed_passthroughs = set()
    removed_connections = set()
    added_connections = set()
    handled_clusters = set()
    for cluster in clusters.values():
        if cluster not in handled_clusters:
            handled_clusters.add(cluster)
            has_onchip_input = False
            has_onchip_output = False
            for c in cluster.conns_in:
                if base_obj(c.pre_obj) not in offchip:
                    has_onchip_input = True
                    break
            for c in cluster.conns_out:
                if base_obj(c.post_obj) not in offchip:
                    has_onchip_output = True
                    break
            no_input = len(cluster.conns_in) == 0
            no_output = len(cluster.conns_out) + len(cluster.probed_objs) == 0

            if (has_onchip_input and has_onchip_output) or no_input or no_output:
                try:
                    new_conns = list(cluster.generate_conns())
                except CycleException:
                    continue

                removed_passthroughs.update(cluster.objs - cluster.probed_objs)
                removed_connections.update(cluster.conns_in | cluster.conns_mid | cluster.conns_out)
                added_connections.update(new_conns)
    return removed_passthroughs, removed_connections, added_connections
