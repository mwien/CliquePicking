#include<iostream>
#include<gmpxx.h>
#include<bitset>
#include<vector>
#include<queue>
#include<unordered_set>
#include<list>
#include<unordered_map>
#include<functional>

typedef mpz_class Z;

using namespace std;
// from: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key/12996028#12996028
unsigned int customhash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

// simple additive hash function for unordered set
size_t us_hash(unordered_set<int> s)
{
    size_t sum = 0;
    for(auto it = s.begin(); it != s.end(); ++it) {
	sum += customhash(*it);
    }
    return sum;
}

// computes the factorial with memoization for better performance
Z fac(int n, vector<Z> &fmemo)
{
    if(fmemo[n] != 0) return fmemo[n];
    if(n == 1) return 1;
    // we define 0! this way for more elegant code
    if(n == 0) return 0;
    
    Z res= fac(n-1, fmemo) * n;
    fmemo[n] = res;
    return res;
}

// find connected component with u in it and store it in newcomp
void dfs(vector<vector<int>> &g, int u, vector<bool> &vis, unordered_set<int> &newcomp)
{
    for(int i = 0; i < g[u].size(); ++i) {
	int v = g[u][i];
	if(!vis[v]) {
	    newcomp.insert(v);
	    vis[v] = true;
	    dfs(g, v, vis, newcomp);
	}
    }
}

// computes a vector of all components of g
void findcomps(vector<vector<int>> &g, int n, vector<unordered_set<int>> &comps)
{
    vector<bool> vis (n);
    for(int i = 0; i < n; ++i) {
	if(!vis[i]) {
	    unordered_set<int> newcomp;
	    newcomp.insert(i);
	    vis[i] = true;
	    dfs(g, i, vis, newcomp);
	    comps.push_back(newcomp);
	}
    }
}

// mp is the mapping to the original vertices
// computes subgraph induced by comp and mapping to orig. vert. for this subgraph
void constructgraph(vector<vector<int>> &g, int n, vector<int> &mp, unordered_set<int> &comp, vector<vector<int>> &newg, vector<int> &newmp)
{
    newg.resize(comp.size());
    unordered_map<int, int> invmp;
    int cnt = 0;
    for(auto it = comp.begin(); it != comp.end(); ++it) {
	newmp.push_back(mp[*it]);
	invmp[*it] = cnt;
	++cnt;
    }
    for(auto it = comp.begin(); it != comp.end(); ++it) {
	int u = *it;
	for(int i = 0; i < g[u].size(); ++i) {
	    int v = g[u][i];
	    if(comp.find(v) != comp.end()) {
		newg[invmp[u]].push_back(invmp[v]);
	    }
	}
    }
}

// computes lbfs order and assigns class label integer to each vertex (new subproblems)
// beg has to be copied because we manipulate it
void lbfs(vector<vector<int>> &g, int n, unordered_set<int> beg, vector<pair<int, int>> &order)
{
    // sequence of sets
    list<int> sigma;
    // pointers to each element in sigma
    vector<list<int>::iterator> elpointer (n);
    // pointers to beg of each set in sigma
    vector<list<int>::iterator> setpointer;
    // the set an element is in
    vector<int> eltoset (n);
    // pointer to new set created when visiting neighbors
    // at beginning first set has no "new set"
    vector<int> newset (1, -1);
    // class label defining the subproblems
    vector<int> number (n, -1);
    // highest label assigned
    int curnum = -1;
    // fill sigma and initialize pointers
    for(int i = 0; i < n; ++i) {
	sigma.push_back(i);
	elpointer[i] = prev(sigma.end());
    }
    // init set pointers
    setpointer.push_back(sigma.begin());
    vector<bool> vis (n);
    while(!sigma.empty()) {
	int u;
	// first draw from beg if nonempty
	if(!beg.empty()) {
	    auto it = beg.begin();
	    u = (*it);
	    beg.erase(it);
	} else { // then from first set
	    u = sigma.front();
	    // assign labels to the set of u if u has no label yet
	    for(auto it = elpointer[u]; it != sigma.end() && number[*it] == -1 && eltoset[*it] == eltoset[u]; ++it) {
		if((*it) == u) ++curnum; 
		number[*it] = curnum;
	    }
	}
	
	sigma.erase(elpointer[u]);
	order.push_back({u, number[u]});
	vis[u] = true;
	// we always take from first set (because beg is a clique)
	setpointer[eltoset[u]] = sigma.begin();
	// number of sets before visiting neighbors of u
	int setcount = setpointer.size();
	for(int i = 0; i < g[u].size(); ++i) {
	    int v = g[u][i];

	    if(vis[v]) continue;
	    int curset = eltoset[v];
	    // newset was already created, i.e. a prev. neighbor was in same set as v
	    if(newset[curset] >= setcount) {
		// make sure pointer of old set is set correctly
		if((*setpointer[curset]) == v) {
		    ++setpointer[curset];
		}
		int ns = newset[curset];
		sigma.erase(elpointer[v]);
		eltoset[v] = ns;
		// insert v at beginning of newset
		setpointer[ns] = sigma.insert(setpointer[ns], v);
		elpointer[v] = setpointer[ns];
	    } else { // we create newset
		newset[curset] = setpointer.size();
		newset.push_back(-1);
		// put v before current set
		setpointer.push_back(sigma.insert(setpointer[curset], v));
		sigma.erase(elpointer[v]);
		eltoset[v] = newset[curset];
		elpointer[v] = setpointer[eltoset[v]];
		setpointer[curset] = next(elpointer[v]);
	    }
	}
    }
}
// computes the clique-tree of g
void findcliques(vector<vector<int>> &g, int n, vector<unordered_set<int>> &cliques, vector<vector<int>> &cltree)
{
    vector<pair<int, int>> order;
    // empty beg, because we just start any LBFS and find cl. tree from PEO
    unordered_set<int> beg;
    lbfs(g, n, beg, order);
    // PEO to cl. tree algorithm from: Computing a Clique Tree with the Algorithm Maximal Label Search
    // by Berry and Simonet --> Algorithm 3
    
    // invorder stores pos of a vertex in the order
    vector<int> invorder (n);
    for(int i = 0; i < order.size(); ++i) {
	invorder[order[i].first] = i;
    }
    // stores clique a vertex is part of
    vector<int> vertextoclique (n, -1);
    // start with empty first clique
    cliques.push_back(unordered_set<int>());
    cltree.push_back(vector<int>());
    // number of created cliques
    int cntcliq = 1;
    for(int i = 0; i < order.size(); ++i) {
	int u = order[i].first;
	int p;
	int cntneigh = 0;
	if(i == 0) {
	    p = 0;
	} else {
	    int k = -1;
	    for(int j = 0; j < g[u].size(); ++j) {
		int v = g[u][j];
		if(invorder[v] < invorder[u]) {
		    // count number of preceding neighbors
		    ++cntneigh;
		    // find prec. neighb. with highest pos. (k is pos. and p corr. clique)
		    if(k < invorder[v]) {
			k = invorder[v];
			p = vertextoclique[v];
		    }
		}
	    }
	}
	// check if u enlargens clique p
	// this simple size check should suffice, as preceding neighbors form clique
	// if size smaller, clearly cannot enlargen this max clique
	// if size the same, preceding neighbors are exactly this clique because
	// prec. neighbors of u are also prec. neighbors of v (highest pos. vertex found above)
	// and p is maximal clique
	if(cntneigh == cliques[p].size()) {
	    vertextoclique[u] = p;
	} else {
	    // start new clique
	    cliques.push_back(unordered_set<int>());
	    for(int j = 0; j < g[u].size(); ++j) {
		int v = g[u][j];
		if(invorder[v] < invorder[u]) {
		    cliques[cntcliq].insert(v);
		}
	    }
	    vertextoclique[u] = cntcliq;
	    // add edge in clique-tree
	    cltree.push_back(vector<int>());
	    cltree[p].push_back(cntcliq);
	    cltree[cntcliq].push_back(p);
	    ++cntcliq;
	}
	cliques[vertextoclique[u]].insert(u);
    }
}

// recursive computation of phi using memoization
// note that the formula we use is more or less the complement of the one in the main paper (both work equally well)
// this recursive formula is presented in the proof of Thm. 4 in the supplementary material
Z recphi(vector<int> &fp, int i, int c, vector<Z> &phimemo, vector<Z> &fmemo)
{
    if(phimemo[i] != 0) return phimemo[i];
    Z sum = fac(c - fp[i], fmemo);
    for(int j = i+1; j < fp.size(); ++j) {
	sum -= fac(fp[j] - fp[i], fmemo) * recphi(fp, j, c, phimemo, fmemo);
    }
    phimemo[i] = sum;
    return sum;
}

// computes the subproblems of g when starting with clique
void findsubproblems(vector<vector<int>> &g, int n, vector<int> &mp, unordered_set<int> &clique, vector<vector<vector<int>>> &newgs, vector<vector<int>> &newmps)
{
    vector<pair<int, int>> order;
    lbfs(g, n, clique, order);
    // vector of sets of vertices with same label
    vector<unordered_set<int>> eqsets;
    for(int i = 0; i < order.size(); ++i) {
	if(order[i].second < 0) continue;
	while(eqsets.size() <= order[i].second) {
	    eqsets.push_back(unordered_set<int>());   
	}
	eqsets[order[i].second].insert(order[i].first);
    }
    // what complicates things, is that the subgraphs induced from the eqsets
    // are not necessarily connected
    // thus, we first construct the induced subgraphs and then find the
    // components of these subgraphs
    for(int i = 0; i < eqsets.size(); ++i) {
	vector<vector<int>> uncong;
	vector<int> unconmp;
	// construct subgraph for each eqset
	constructgraph(g, n, mp, eqsets[i], uncong, unconmp);
	vector<unordered_set<int>> comps;
	findcomps(uncong, uncong.size(), comps);
	for(int j = 0; j < comps.size(); ++j) {
	    vector<vector<int>> newg;
	    vector<int> newmp;
	    // for each component construct the graph and push it to newgs
	    constructgraph(uncong, uncong.size(), unconmp, comps[j], newg, newmp);
	    newgs.push_back(newg);
	    newmps.push_back(newmp);
	}
    }
}

// main function, computes the number of MAOs for UCCG G
// mp stores mapping to original vertices (necessary for memoization)
Z solve(vector<vector<int>> &g, int n, vector<int> &mp, unordered_map<unordered_set<int>, Z, decltype(us_hash) *> &memo, vector<Z> &fmemo)
{
    unordered_set<int> hs;
    for(int i = 0; i < n; ++i) {
	hs.insert(mp[i]);
    }

    if(memo.find(hs) != memo.end()) {
	return memo[hs];
    }

    // first, compute clique-tree
    vector<unordered_set<int>> cliques;
    vector<vector<int>> cltree;
    findcliques(g, n, cliques, cltree);
    int c = cliques.size();

    // go through cliques in BFS order starting at arbitrary node
    Z sum = 0;
    vector<bool> vis (c);
    queue<int> q;
    q.push(0);
    vis[0] = true;
    vector<int> pred (c);
    for(int i = 0; i < c; ++i) {
	pred[i] = -1;
    }
    while(!q.empty()) {
	int u = q.front(); q.pop();
	for(int j = 0; j < cltree[u].size(); ++j) {
	    int v = cltree[u][j];
	    if(!vis[v]) {
		q.push(v);
		vis[v] = true;
		pred[v] = u;
	    }
	}
	// calculate phi
	// go backwards from u to the root 0 to construct fp
	int idx = u;
	int lcliq = -1;
	vector<int> fp;
	while(pred[idx] != -1) {
	    idx = pred[idx];
	    int cutsz = 0;
	    int lcliqcutsz = 0;
	    for(auto it = cliques[idx].begin(); it != cliques[idx].end(); ++it) {
		int x = (*it);
		// calculate size of cut between idx and u
		if(cliques[u].find(x) != cliques[u].end()) {
		    ++cutsz;
		}
		// calculate size of cut between idx and lidx
		if(lcliq != -1 && cliques[lcliq].find(x) != cliques[lcliq].end()) {
		    ++lcliqcutsz;
		}
	    }
	    lcliq = idx;
	    if(cutsz == 0) break;
	    // if lastcut were strictly greater, u is not in bouquet defined by cut between idx and lidx
	    if(cutsz >= lcliqcutsz && (fp.empty() || cutsz < fp.back())) {
		fp.push_back(cutsz);
	    }
	}
	fp.push_back(0);
	reverse(fp.begin(), fp.end());
	vector<Z> phimemo (fp.size());
	Z nos = recphi(fp, 0, cliques[u].size(), phimemo, fmemo);
	// compute subproblems
	vector<vector<vector<int>>> newgs;
	vector<vector<int>> newmps;
	findsubproblems(g, n, mp, cliques[u], newgs, newmps);
	Z cnt = 1;
	for(int i = 0; i < newgs.size(); ++i) {
	    // #MAOs are multiplied as the subproblems are independent
	    cnt *= solve(newgs[i], newgs[i].size(), newmps[i], memo, fmemo);
	}
	cnt *= nos;
	sum += cnt;
    }
    memo[hs] = sum;
    return sum;
}

int main()
{
    // read from adjacency matrix
    // int n;
    // cin >> n;
    // vector<vector<int>> g (n);
    // for(int i = 0; i < n; ++i) {
    // 	for(int j = 0; j < n; ++j) {
    // 	    int x;
    // 	    cin >> x;
    // 	    if(x == 1) {
    // 		g[i].push_back(j);
    // 	    }
    // 	}
    // }

    // read from adjacency list
    int n, m;
    cin >> n >> m;
    vector<vector<int>> g (n);
    for(int i = 0; i < m; ++i) {
    	int a, b;
    	cin >> a >> b;
    	g[a-1].push_back(b-1);
    	g[b-1].push_back(a-1);
    }

    vector<Z> fmemo (n+1);

    vector<unordered_set<int>> comps;
    findcomps(g, n, comps);

    vector<int> mp;
    for(int i = 0; i < n; ++i) {
	mp.push_back(i);
    }

    Z res = 1;
    for(int i = 0; i < comps.size(); ++i) {
	unordered_map<unordered_set<int>, Z, decltype(us_hash) *> memo (100000, us_hash);
	vector<vector<int>> newg;
	vector<int> newmp;
	constructgraph(g, n, mp, comps[i], newg, newmp);
     	res *= solve(newg, newg.size(), newmp, memo, fmemo);
	// cout << "m " << memo.size() << endl;
    }
    cout << res << endl;
}
