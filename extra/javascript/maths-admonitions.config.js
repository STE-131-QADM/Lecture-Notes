window.MkDocsMathsAdmonitionsConfig = {
    locale: "en",

    numbering: {
        enabled: true,
        start: 1,
        separator: "."
    },

    // Number only mathematical statements.
    defaults: {
        numbered: false
    },

    admonitions: {
        theorem: {
            numbered: true
        },
        definition: {
            numbered: true
        },
        proposition: {
            numbered: true
        },
        lemma: {
            numbered: true
        },
        corollary: {
            numbered: true
        },
        proof: {
            numbered: false
        },
        example: {
            numbered: false
        },
        remark: {
            numbered: false
        }
    }
};
