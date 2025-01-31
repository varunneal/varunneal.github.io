An LLM [doesn't care where its CoT knowledge came from](https://github.com/getasterisk/deepclaude?tab=readme-ov-file#why-r1--claude). Therefore, its CoT should be hacked to be as helpful as possible, even if the LLM itself cannot generate it.

Consider this deepseek-r1:14b CoT:  
```  
<think>  
Okay, so I need to figure out how many "rs" are in the word "strawberry." Hmm, let's start by  
spelling it out to make sure I have it right. S-T-R-A-W-B-E-R-R-Y. Wait, no, that can't be right  
because when I think of "strawberry," I know it has two r sounds, but maybe more letters.

Let me write it down properly: S-T-R-A-W-B-E-R-R-Y? No, wait, the correct spelling is actually  
S-T-R-A-W-B-E-R-R-Y. So that's how it goes. Now, I'm trying to count the number of times the letter  
'r' appears in this word.

Looking at each letter one by one: S, T, R, A, W, B, E, R, R, Y. Wait, so after the first R, there  
are two more Rs? Let me check again. After the initial R (the third letter), I think there's another  
R later on. So maybe that's it.

Wait, no, let me make sure. The word is S-T-R-A-W-B-E-R-R-Y. So starting from the beginning: S, T,  
then R at position 3, then A, W, B, E, and then R again at position 8, followed by another R at  
position 9 before ending with Y.

So that means there are three Rs in "strawberry." Wait, no, that doesn't sound right because when I  
think of it, the word has two 'r' sounds. Let me check a reliable source or maybe use another method  
to confirm.

Alternatively, maybe I'm overcomplicating this. The word is spelled S-T-R-A-W-B-E-R-R-Y. So from  
position 3: R, then after E comes R again, and another R? Wait no, that's not correct because the  
word should be S-T-R-A-W-B-E-R-R-Y, which actually has two Rs at the end.

Wait, perhaps I'm confusing it with "strawberry" being spelled as S-T-R-A-W-B-E-R-R-Y. Let me count  
again: 1 (R), then another R later on—so that's two Rs in total.  
</think>

The word "strawberry" contains two 'r's.

**Answer:** There are 2 'rs' in "strawberry."  
```

Instead of CoT, let's suppose we fine-tuned the LLM to use Oracles via tool-calling:  
```  
<oracle>  
I need to figure out how many "rs" are in the word "strawberry." A reliable source tells me that there are <python>'strawberry'.count('r')</python>  
```  
before the next token is generated, the `<python>` statement gets automatically executed, so the CoT finishes as:  
```  
<oracle>  
I need to figure out how many "rs" are in the word "strawberry." A reliable source tells me that there are 3 "r"s in "strawberry."  
</oracle>
The word "strawberry contains three 'r's.  
```

I think it is likely that such Oracle-like use of tool calling will be an emergent property of equipping current CoT SL with interpreters and other APIs. 